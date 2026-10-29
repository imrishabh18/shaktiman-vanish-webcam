"""Exercise the camera state machine without opening a physical camera."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from shaktiman import app
from tests.test_gesture import hand_points, landmarks


class ShaktimaanCameraTests(unittest.TestCase):
    def run_video(self, times, keys, scores, empty_mask=False, background_people=False, setup_ui=False):
        clock = [0.0]
        frames = [np.full((48, 64, 3), 20 + i * 12, np.uint8) for i in range(len(times))]
        samples = iter(zip(times, frames))

        def read():
            clock[0], frame = next(samples)
            return True, frame

        capture = Mock()
        capture.read.side_effect = read
        recognizer = Mock(last_hand_count=1)
        recognizer.recognize_point_up.side_effect = scores
        segmenter = Mock()
        audio = Mock()
        camera = Mock(width=64, height=48)
        args = SimpleNamespace(
            camera=0, width=64, height=48, fps=30, fullscreen=False,
            no_mirror=True, virtual_camera=True, virtual_backend=None,
            virtual_device=None, gesture_debug=False,
        )
        mask = np.zeros((48, 64), np.float32)
        if not empty_mask:
            mask[8:47, 19:45] = 1
        displayed = []
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "create_audio", return_value=audio))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            mocks.enter_context(patch.object(app, "create_tasks", return_value=(segmenter, recognizer)))
            mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=camera))
            if isinstance(background_people, list):
                mocks.enter_context(patch.object(app, "calibration_person_present", side_effect=background_people))
            else:
                mocks.enter_context(patch.object(app, "calibration_person_present", return_value=background_people))
            if not setup_ui:
                mocks.enter_context(patch.object(app, "calibration_preview", side_effect=lambda frame, *_: frame))
            segmentation = mocks.enter_context(patch.object(app, "person_mask", return_value=mask))
            snapshot = mocks.enter_context(patch.object(app, "capture_shaktimaan", wraps=app.capture_shaktimaan))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=lambda: clock[0]))
            mocks.enter_context(patch.object(app.cv2, "namedWindow"))
            mocks.enter_context(patch.object(app.cv2, "resizeWindow"))
            mocks.enter_context(patch.object(app.cv2, "destroyAllWindows"))
            mocks.enter_context(patch.object(app.cv2, "imshow", side_effect=lambda _, frame: displayed.append(frame.copy())))
            mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=keys))
            mocks.enter_context(patch("builtins.print"))
            self.assertEqual(app.run_camera(args, ExitStack()), 0)
        capture.release.assert_called_once()
        segmenter.close.assert_called_once()
        recognizer.close.assert_called_once()
        camera.close.assert_called_once()
        audio.close.assert_called_once()
        self.assertEqual(audio.play.call_count, snapshot.call_count if not empty_mask else 0)
        if ord("r") in keys or ord(" ") in keys:
            if snapshot.call_count:
                self.assertGreaterEqual(audio.stop.call_count, 1)
        if setup_ui:
            for sent, preview, original in zip(camera.send.call_args_list, displayed, frames):
                np.testing.assert_array_equal(sent.args[0], original)
                self.assertFalse(np.array_equal(preview, original))
        else:
            for sent, preview in zip(camera.send.call_args_list, displayed):
                np.testing.assert_array_equal(sent.args[0], preview)
        self.assertEqual(camera.send.call_count, len(displayed))
        return frames, displayed, snapshot, segmentation

    def test_index_hold_spins_once_and_leaves_empty_background(self):
        frames, displayed, snapshot, segmentation = self.run_video(
            [0.1, 4.1, 4.2, 4.35, 4.5, 4.6, 5.1, 7.1, 7.2],
            [0] * 8 + [ord("q")], [1.0] * 4,
        )
        snapshot.assert_called_once()
        segmentation.assert_called_once()
        np.testing.assert_array_equal(displayed[4], frames[4])
        self.assertTrue(np.any(displayed[6] != frames[1]))
        np.testing.assert_array_equal(displayed[7], frames[1])
        np.testing.assert_array_equal(displayed[8], frames[1])

    def test_return_requires_release_and_new_hold_before_second_spin(self):
        _, _, snapshot, segmentation = self.run_video(
            [0.1, 4.1, 4.2, 4.35, 4.5, 4.6, 7.1, 7.2,
             7.3, 7.45, 7.6, 7.75, 7.9, 8.05, 8.2, 8.3],
            [0] * 7 + [ord("r")] + [0] * 7 + [ord("q")],
            [1.0] * 7 + [0.0] + [1.0] * 4,
        )
        self.assertEqual(snapshot.call_count, 2)
        self.assertEqual(segmentation.call_count, 2)

    def test_v_starts_same_effect_without_pointing(self):
        frames, displayed, snapshot, _ = self.run_video(
            [0.1, 4.1, 4.2, 4.35, 7.0],
            [0, 0, ord("v"), 0, ord("q")], [0.0],
        )
        snapshot.assert_called_once()
        np.testing.assert_array_equal(displayed[-1], frames[1])

    def test_missing_person_cancels_without_hiding_feed(self):
        frames, displayed, snapshot, _ = self.run_video(
            [0.1, 4.1, 4.2, 4.35, 4.5, 4.6, 4.75, 4.9],
            [0] * 7 + [ord("q")], [1.0] * 6, empty_mask=True,
        )
        snapshot.assert_called_once()
        np.testing.assert_array_equal(displayed[-1], frames[-1])

    def test_unrotated_detector_accepts_either_visible_hand(self):
        primary = Mock()
        primary.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[landmarks(hand_points((False,) * 4)), landmarks(hand_points())],
        )
        detector = app.PointUpHandLandmarker(primary)
        image = SimpleNamespace(width=640, height=480)
        self.assertGreater(detector.recognize_point_up(image, 120), 0.88)
        primary.detect_for_video.assert_called_once_with(image, 120)
        self.assertEqual(detector.last_hand_count, 2)
        detector.close()
        primary.close.assert_called_once()

    def test_person_in_view_never_becomes_the_saved_background(self):
        frames, displayed, snapshot, segmentation = self.run_video(
            [0.1, 4.1, 8.2, 12.3],
            [0, ord("v"), ord("r"), ord("q")], [], background_people=True,
        )
        snapshot.assert_not_called()
        segmentation.assert_not_called()
        np.testing.assert_array_equal(displayed[-1], frames[-1])

    def test_leaving_the_room_allows_capture_then_vanish_stays_until_return(self):
        frames, displayed, snapshot, _ = self.run_video(
            [0.1, 4.1, 4.3, 8.3, 8.5, 8.7, 8.9, 9.1, 11.7, 12.0, 13.0],
            [0] * 10 + [ord("q")], [1.0] * 4,
            background_people=[True, True, False, False],
        )
        snapshot.assert_called_once()
        # Frame 3 is the checked empty room; later live frames contain the user.
        for index in (8, 9, 10):
            np.testing.assert_array_equal(displayed[index], frames[3])

    def test_setup_instructions_are_only_on_the_local_preview(self):
        self.run_video([0.1, 4.1, 4.2], [0, 0, ord("q")], [], background_people=True, setup_ui=True)

    def test_recapture_discards_old_background_and_r_cannot_skip_setup(self):
        frames, displayed, snapshot, _ = self.run_video(
            [0.1, 4.1, 4.2, 4.4, 7.0, 7.2, 7.4, 8.0, 12.4],
            [0, 0, ord("v"), 0, 0, ord(" "), 0, ord("r"), ord("q")], [0.0],
            background_people=[False, False, True, True, True],
        )
        snapshot.assert_called_once()
        np.testing.assert_array_equal(displayed[-1], frames[-1])


if __name__ == "__main__":
    unittest.main()
