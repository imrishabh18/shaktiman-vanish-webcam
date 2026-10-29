"""Webcam capture, clean-room setup, gesture tracking, and virtual output."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import os
from pathlib import Path
import platform
import ssl
import sys
import time
import urllib.error
import urllib.request

APP_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = APP_DIR / "models"
os.environ.setdefault("MPLCONFIGDIR", str(APP_DIR / ".cache" / "matplotlib"))

import certifi
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .background import CleanBackgroundCapture, calibration_preview, mask_has_person
from .audio import create_audio
from .cameras import mac_camera_devices, preferred_webcam
from .effect import SHAKTIMAAN_SECONDS, capture_shaktimaan, render_shaktimaan
from .gesture import PointUpTrigger, pointing_up_score

WINDOW_NAME = "Shaktiman Vanish"
SEGMENTER_MODEL = MODEL_DIR / "selfie_segmenter_landscape.tflite"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
MODEL_URLS = {
    SEGMENTER_MODEL: "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite",
    HAND_MODEL: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}
GESTURE_INTERVAL_SECONDS = 0.08

@dataclass
class AppState:
    phase: str = "calibrating"
    vanish_started: float | None = None
    last_gesture_check: float = 0.0
    fullscreen: bool = False

def positive_fps(value: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or not 1 <= fps <= 120:
        raise argparse.ArgumentTypeError("FPS must be between 1 and 120.")
    return fps


def music_volume(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("Music volume must be between 0 and 1.")
    return number


def music_start(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Music start must be nonnegative and finite.")
    return number

def send_virtual_frame(camera, frame: np.ndarray) -> None:
    if frame.shape[:2] != (camera.height, camera.width):
        frame = cv2.resize(frame, (camera.width, camera.height), interpolation=cv2.INTER_LINEAR)
    camera.send(np.ascontiguousarray(frame))

def open_camera(index: int | None, width: int, height: int) -> cv2.VideoCapture:
    if index is None and platform.system() == "Darwin":
        device = preferred_webcam(mac_camera_devices())
        index = device.index
        print(f"[Webcam] Selected {device.name} (camera {index}).", flush=True)
    if index is None:
        index = 0
    if platform.system() == "Windows":
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
    elif platform.system() == "Darwin":
        capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        if platform.system() == "Darwin":
            raise RuntimeError(
                f"Cannot open webcam {index}. If macOS requested camera access, "
                "allow it and reopen start.command from Finder or Terminal. "
                "Check System Settings > Privacy & Security > Camera for Terminal, "
                "and close any other app using the camera."
            )
        raise RuntimeError(
            f"Camera {index} is unavailable. Close other camera apps or try --camera 1."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture

def person_mask(
    segmenter: vision.ImageSegmenter,
    image: mp.Image,
    timestamp_ms: int,
    width: int,
    height: int,
) -> np.ndarray:
    result = segmenter.segment_for_video(image, timestamp_ms)
    if not result.confidence_masks:
        return np.zeros((height, width), dtype=np.float32)
    raw = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
    resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
    resized = cv2.GaussianBlur(resized, (0, 0), sigmaX=2.2, sigmaY=2.2)
    return np.clip((resized - 0.12) / 0.68, 0.0, 1.0)

def calibration_person_present(segmenter, image: mp.Image, timestamp_ms: int, width: int, height: int) -> bool:
    result = segmenter.segment_for_video(image, timestamp_ms)
    if not result.confidence_masks:
        return True  # An unavailable mask must not count as an empty room.
    return mask_has_person(np.squeeze(result.confidence_masks[0].numpy_view()))

class PointUpHandLandmarker:
    """Detect a single upward-pointing index in the unrotated camera frame."""

    def __init__(self, primary):
        self.primary = primary
        self.last_hand_count = 0

    def recognize_point_up(self, image: mp.Image, timestamp_ms: int) -> float:
        result = self.primary.detect_for_video(image, timestamp_ms)
        self.last_hand_count = len(result.hand_landmarks)
        return max((pointing_up_score(hand, image.width, image.height)
                    for hand in result.hand_landmarks), default=0.0)

    def close(self) -> None:
        self.primary.close()

def set_fullscreen(enabled: bool) -> None:
    mode = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Point your index finger up, spin, and vanish.")
    parser.add_argument("--camera", type=int, help="Camera index; macOS prefers the built-in webcam")
    parser.add_argument("--list-cameras", action="store_true", help="List camera names on macOS")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=positive_fps, default=30.0)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--virtual-camera", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--virtual-backend", choices=("obs", "unitycapture", "v4l2loopback"))
    parser.add_argument("--virtual-device")
    parser.add_argument("--gesture-debug", action="store_true")
    parser.add_argument("--music", type=Path, help="Music clip; defaults to music.wav or music.mp3 in this folder")
    parser.add_argument("--music-start", type=music_start, default=0, help="Start time in the music file, in seconds")
    parser.add_argument("--music-volume", type=music_volume, default=0.5, help="Music volume, 0 to 1")
    parser.add_argument("--no-music", action="store_true", help="Mute the animation cue")
    parser.add_argument("--meeting-audio", action="store_true", help="Mix microphone and music into BlackHole for calls")
    parser.add_argument("--audio-output", help="Virtual output name/index; defaults to BlackHole for meeting audio")
    parser.add_argument("--microphone", help="Physical microphone name/index for meeting audio")
    parser.add_argument("--list-audio-devices", action="store_true")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def download_models() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for destination, url in MODEL_URLS.items():
        if destination.is_file() and destination.stat().st_size > 100_000:
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"[Setup] Downloading {destination.name}...", flush=True)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Shaktiman-Vanish/1.0"})
            with urllib.request.urlopen(request, timeout=60,
                                        context=ssl.create_default_context(cafile=certifi.where())) as response:
                with temporary.open("wb") as output:
                    while chunk := response.read(256 * 1024):
                        output.write(chunk)
            if temporary.stat().st_size < 100_000:
                raise OSError("The downloaded model is incomplete.")
            temporary.replace(destination)
        except (OSError, urllib.error.URLError) as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Model download failed. Check your internet connection and restart the app.") from error


def create_tasks():
    segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    ))
    try:
        hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.45,
        ))
    except Exception:
        segmenter.close()
        raise
    return segmenter, PointUpHandLandmarker(hands)


def open_virtual_camera(args: argparse.Namespace, width: int, height: int):
    if args.virtual_camera is False:
        return None
    try:
        import pyvirtualcam
        camera = pyvirtualcam.Camera(
            width=width, height=height, fps=args.fps, fmt=pyvirtualcam.PixelFormat.BGR,
            backend=args.virtual_backend, device=args.virtual_device,
        )
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        message = (
            "Virtual camera unavailable. Install OBS Studio, start and stop its virtual camera once, "
            f"then quit OBS and restart this app. Details: {error}"
        )
        if args.virtual_camera is True or args.virtual_backend or args.virtual_device:
            raise RuntimeError(message) from error
        print(f"[Camera] {message}\nContinuing with local preview.")
        return None
    print(f"[Camera] Sending video to {camera.device}.", flush=True)
    return camera


def run_camera(args: argparse.Namespace, resources: ExitStack | None = None) -> int:
    with resources or ExitStack() as resources:
        download_models()
        audio = create_audio(args, APP_DIR)
        if audio is not None:
            resources.callback(audio.close)
        capture = open_camera(args.camera, args.width, args.height)
        resources.callback(capture.release)
        capture.set(cv2.CAP_PROP_FPS, args.fps)
        segmenter, recognizer = create_tasks()
        resources.callback(segmenter.close)
        resources.callback(recognizer.close)
        resources.callback(cv2.destroyAllWindows)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, args.width, args.height)
        state = AppState(fullscreen=args.fullscreen)
        if state.fullscreen:
            set_fullscreen(True)

        calibration = CleanBackgroundCapture()
        trigger = PointUpTrigger()
        background = snapshot = None
        camera = None
        output_initialized = False
        last_background_check = float("-inf")
        last_calibration_message = None
        ready_notice_until = last_gesture_report = 0.0
        started = time.perf_counter()
        print("[Setup] Step fully out of view. Return only after BACKGROUND READY appears.")
        print("[Controls] Index finger UP to vanish. V test, R return, SPACE recapture, F fullscreen, Q quit.")

        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("The webcam stopped providing video. Close other camera apps and restart.")
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            if not output_initialized:
                camera = open_virtual_camera(args, width, height)
                output_initialized = True
                if camera is not None:
                    resources.callback(camera.close)
            now = time.perf_counter()
            timestamp = int((now - started) * 1000)
            image = None

            if state.phase == "calibrating":
                if now - last_background_check >= 0.14:
                    last_background_check = now
                    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    present = calibration_person_present(segmenter, image, timestamp, width, height)
                    clean = calibration.observe(frame, present, now)
                    status = (calibration.status, calibration.remaining)
                    if status != last_calibration_message:
                        if calibration.status == "waiting":
                            print("[Setup] Person detected. Step fully out of view.")
                        elif calibration.status == "capturing":
                            print(f"[Setup] Empty room: {calibration.remaining}s. Stay out of view.")
                        last_calibration_message = status
                    if clean is not None:
                        background = clean
                        state.phase = "ready"
                        ready_notice_until = now + 2.5
                        print("[Setup] BACKGROUND READY - STEP BACK IN.")
                display = frame
            elif state.phase == "vanished":
                display = background
            else:
                if state.phase == "ready" and now - state.last_gesture_check >= GESTURE_INTERVAL_SECONDS:
                    state.last_gesture_check = now
                    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    score = recognizer.recognize_point_up(image, timestamp)
                    if args.gesture_debug and now - last_gesture_report >= 1:
                        print(f"[Gesture] hands={recognizer.last_hand_count}, index_up={score:.2f}")
                        last_gesture_report = now
                    if trigger.update(score >= 0.62, now):
                        state.phase = "spinning"
                        state.vanish_started = time.perf_counter()
                        snapshot = None

                if state.phase == "spinning" and snapshot is None:
                    if image is None:
                        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    mask = person_mask(segmenter, image, timestamp, width, height)
                    snapshot = capture_shaktimaan(frame, mask)
                    if snapshot is None:
                        state.phase = "ready"
                        state.vanish_started = None
                        trigger.reset(require_release=True)
                        print("[Effect] No person found. Lower your hand, step into view, and try again.")
                    else:
                        # Start both clocks after segmentation, keeping the cue
                        # aligned with the first visible animation frame.
                        state.vanish_started = time.perf_counter()
                        if audio is not None:
                            audio.play()
                if state.phase == "spinning":
                    age = max(0.0, time.perf_counter() - state.vanish_started)
                    display = render_shaktimaan(background, snapshot, age)
                    if age >= SHAKTIMAAN_SECONDS:
                        state.phase = "vanished"
                        display = background
                else:
                    display = frame

            if camera is not None:
                send_virtual_frame(camera, display)
            preview = display
            if state.phase == "calibrating":
                preview = calibration_preview(display, calibration.status, calibration.remaining)
            elif state.phase == "ready" and now < ready_notice_until:
                preview = calibration_preview(display, "ready", 0)
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if audio is not None:
                    audio.stop()
                state.phase = "calibrating"
                state.vanish_started = None
                background = snapshot = None
                trigger.reset()
                calibration.reset()
                last_background_check = float("-inf")
                last_calibration_message = None
                ready_notice_until = 0
                print("[Setup] Step fully out of view and wait for BACKGROUND READY.")
            elif key == ord("r") and background is not None:
                if audio is not None:
                    audio.stop()
                state.phase = "ready"
                state.vanish_started = None
                snapshot = None
                trigger.reset(require_release=True)
            elif key == ord("v") and state.phase == "ready" and background is not None:
                state.phase = "spinning"
                state.vanish_started = time.perf_counter()
                snapshot = None
                trigger.reset(require_release=True)
            elif key == ord("f"):
                state.fullscreen = not state.fullscreen
                set_fullscreen(state.fullscreen)
            if camera is not None:
                camera.sleep_until_next_frame()
    return 0


def run() -> int:
    args = parse_args()
    if args.list_audio_devices:
        import sounddevice as sd
        for index, device in enumerate(sd.query_devices()):
            print(f"{index}: {device['name']} (inputs={device['max_input_channels']}, outputs={device['max_output_channels']})")
        return 0
    if args.list_cameras:
        if platform.system() != "Darwin":
            raise RuntimeError("Camera name listing is available on macOS. Use --camera INDEX on other systems.")
        for device in mac_camera_devices():
            print(f"{device.index}: {device.name}" + (" [built-in webcam]" if device.built_in else ""))
        return 0
    if args.self_test:
        download_models()
        segmenter, hands = create_tasks()
        try:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((240, 320, 3), np.uint8))
            hands.recognize_point_up(image, 1)
            person_mask(segmenter, image, 1, 320, 240)
        finally:
            hands.close()
            segmenter.close()
        print("[Setup] Hand tracking and segmentation models passed their self-test.")
        return 0
    return run_camera(args)


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"[Shaktiman] {error}", file=sys.stderr)
        return 1
