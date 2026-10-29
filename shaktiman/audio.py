"""Preloaded effect music, optionally mixed with the mic for meeting software."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 48_000
CUE_SECONDS = 3.0


def load_music(path: Path, start: float = 0, volume: float = 0.5) -> np.ndarray:
    """Decode a short cue before the webcam loop; callbacks never access files."""
    if not np.isfinite(start) or start < 0 or not np.isfinite(volume) or not 0 <= volume <= 1:
        raise ValueError("Music start must be nonnegative and volume must be between 0 and 1.")
    with sf.SoundFile(path) as source:
        offset = round(start * source.samplerate)
        if offset >= source.frames:
            raise ValueError("The music start time is past the end of the file.")
        source.seek(offset)
        samples = source.read(round(CUE_SECONDS * source.samplerate), dtype="float32", always_2d=True)
        rate = source.samplerate
    if not len(samples):
        raise ValueError("The music clip is empty.")
    if samples.shape[1] == 1:
        samples = np.repeat(samples, 2, axis=1)
    elif samples.shape[1] > 2:
        samples = np.repeat(samples.mean(axis=1, keepdims=True), 2, axis=1)
    if rate != SAMPLE_RATE:
        count = max(1, round(len(samples) * SAMPLE_RATE / rate))
        positions = np.arange(count) * rate / SAMPLE_RATE
        samples = np.column_stack([np.interp(positions, np.arange(len(samples)), channel)
                                   for channel in samples.T]).astype(np.float32)
    samples = np.nan_to_num(samples).astype(np.float32) * volume
    np.clip(samples, -1, 1, out=samples)
    attack = min(len(samples), round(SAMPLE_RATE * 0.008))
    release = min(len(samples), round(SAMPLE_RATE * 0.045))
    samples[:attack] *= np.linspace(0, 1, attack, dtype=np.float32)[:, None]
    samples[-release:] *= np.linspace(1, 0, release, dtype=np.float32)[:, None]
    return np.ascontiguousarray(samples)


def _device_index(devices, selection, direction: str) -> int:
    key = f"max_{direction}_channels"
    if isinstance(selection, str) and selection.isdecimal():
        selection = int(selection)
    if isinstance(selection, int):
        if 0 <= selection < len(devices) and devices[selection][key] > 0:
            return selection
    else:
        wanted = str(selection).casefold()
        candidates = [(i, d) for i, d in enumerate(devices) if d[key] > 0]
        for exact in (True, False):
            for index, device in candidates:
                name = device["name"].casefold()
                matches = name == wanted if exact else wanted in name
                if matches:
                    return index
    raise ValueError(f"No {direction} audio device matches {selection!r}.")


def meeting_devices(backend=sd, microphone=None, output=None) -> tuple[int, int]:
    devices = backend.query_devices()
    if output is None:
        output = next((i for i, d in enumerate(devices)
                       if "blackhole" in d["name"].casefold() and d["max_output_channels"] >= 2), None)
        if output is None:
            raise RuntimeError(
                "Meeting audio needs a virtual microphone. Install BlackHole 2ch on Mac, "
                "then start with --meeting-audio and choose BlackHole 2ch as the meeting microphone."
            )
    output_index = _device_index(devices, output, "output")
    if microphone is None:
        microphone = int(backend.default.device[0])
        # Do not feed the virtual microphone back into its own output.
        if not 0 <= microphone < len(devices) or microphone == output_index or "blackhole" in devices[microphone]["name"].casefold():
            microphone = next((i for i, d in enumerate(devices)
                               if d["max_input_channels"] > 0 and i != output_index
                               and "blackhole" not in d["name"].casefold()), -1)
    input_index = _device_index(devices, microphone, "input")
    if input_index == output_index:
        raise ValueError("Choose your physical microphone as --microphone, not the virtual output device.")
    return input_index, output_index


@dataclass
class _Playback:
    position: int = 0


class EffectAudio:
    """One persistent audio stream; play/stop only switch a small playback state."""

    def __init__(self, samples: np.ndarray, *, meeting: bool = False,
                 microphone=None, output=None, backend=sd):
        self.samples = samples
        self._playback = None
        self._stream = None
        if meeting:
            inputs, outputs = meeting_devices(backend, microphone, output)
            self._stream = backend.Stream(
                device=(inputs, outputs), channels=(1, 2), samplerate=SAMPLE_RATE,
                dtype="float32", latency="low", callback=self._meeting_callback,
            )
        else:
            if output is not None:
                output = _device_index(backend.query_devices(), output, "output")
            self._stream = backend.OutputStream(
                device=output, channels=2, samplerate=SAMPLE_RATE,
                dtype="float32", latency="low", callback=self._output_callback,
            )
        try:
            self._stream.start()
        except Exception:
            self._stream.close()
            self._stream = None
            raise

    def play(self) -> None:
        if len(self.samples):
            self._playback = _Playback()

    def stop(self) -> None:
        # Stop the cue, not the microphone stream used by the call.
        self._playback = None

    def _mix_cue(self, outdata, frames):
        playback = self._playback
        if playback is None:
            return
        count = min(frames, len(self.samples) - playback.position)
        outdata[:count] += self.samples[playback.position:playback.position + count]
        playback.position += count
        if playback.position >= len(self.samples) and self._playback is playback:
            self._playback = None

    def _output_callback(self, outdata, frames, timing, status):
        outdata.fill(0)
        self._mix_cue(outdata, frames)
        np.clip(outdata, -1, 1, out=outdata)

    def _meeting_callback(self, indata, outdata, frames, timing, status):
        outdata[:] = indata[:, :1]  # Mono voice to both output channels.
        self._mix_cue(outdata, frames)
        np.clip(outdata, -1, 1, out=outdata)

    def close(self) -> None:
        self.stop()
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None


def create_audio(args, root: Path) -> EffectAudio | None:
    meeting = getattr(args, "meeting_audio", False)
    explicit = getattr(args, "music", None)
    path = Path(explicit).expanduser() if explicit else next(
        (root / name for name in ("music.wav", "music.mp3") if (root / name).is_file()), None,
    )
    muted = getattr(args, "no_music", False)
    if (path is None or muted) and not meeting:
        return None
    try:
        samples = (load_music(path, getattr(args, "music_start", 0), getattr(args, "music_volume", 0.5))
                   if path is not None and not muted else np.zeros((0, 2), np.float32))
        audio = EffectAudio(samples, meeting=meeting, microphone=getattr(args, "microphone", None),
                            output=getattr(args, "audio_output", None))
    except (OSError, RuntimeError, ValueError, sd.PortAudioError) as error:
        if explicit or meeting:
            raise RuntimeError(f"Audio setup failed: {error}") from error
        print(f"[Music] Audio unavailable; continuing with video: {error}")
        return None
    if path is not None and not muted:
        print(f"[Music] Ready: {path.name}. Plays once when the animation starts.")
    if meeting:
        print("[Audio] Voice and music are routed to the virtual microphone. Select it in Meet/Zoom.")
    return audio
