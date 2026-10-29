import unittest

import cv2
import numpy as np

from shaktiman.background import CleanBackgroundCapture, calibration_preview, mask_has_person


class PersonPresenceTests(unittest.TestCase):
    def test_person_is_rejected_but_empty_room_and_small_noise_are_allowed(self):
        mask = np.zeros((180, 320), np.float32)
        self.assertFalse(mask_has_person(mask))
        mask[10, 20] = 1
        mask[100, 200] = 1
        self.assertFalse(mask_has_person(mask))
        cv2.ellipse(mask, (160, 110), (30, 65), 0, 0, 360, 0.95, -1)
        self.assertTrue(mask_has_person(mask))

    def test_room_edge_and_bedding_artifacts_do_not_block_capture(self):
        mask = np.zeros((180, 320), np.float32)
        mask[155:, 48:] = 0.99
        self.assertFalse(mask_has_person(mask))
        mask[:] = 0
        mask[:, 303:] = 0.99
        self.assertFalse(mask_has_person(mask))
        # A smaller upright person in the room still blocks capture.
        mask[:] = 0
        mask[80:140, 140:153] = 0.99
        self.assertTrue(mask_has_person(mask))

    def test_invalid_or_unavailable_mask_cannot_allow_capture(self):
        for mask in (np.zeros((0, 0)), np.zeros((20,)), np.full((20, 20), np.nan)):
            self.assertTrue(mask_has_person(mask))


class CleanBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.empty = np.full((30, 40, 3), 35, np.uint8)
        self.person = self.empty.copy()
        self.person[4:30, 10:30] = 170

    def test_person_can_never_be_saved_even_after_timeout(self):
        capture = CleanBackgroundCapture()
        for now in (0, 4, 10, 100):
            self.assertIsNone(capture.observe(self.person, True, now))
            self.assertEqual(capture.status, "waiting")
            self.assertFalse(capture.samples)

    def test_only_empty_frames_are_used_after_person_leaves(self):
        capture = CleanBackgroundCapture()
        capture.observe(self.person, True, 0)
        self.assertIsNone(capture.observe(self.empty, False, 1))
        self.assertIsNone(capture.observe(self.empty, False, 4.5))
        background = capture.observe(self.empty, False, 5)
        self.assertEqual(capture.status, "ready")
        np.testing.assert_array_equal(background, self.empty)
        self.empty[:] = 0
        self.assertTrue(np.all(background == 35))

    def test_reentering_during_countdown_discards_samples_and_restarts(self):
        capture = CleanBackgroundCapture()
        capture.observe(self.empty, False, 0)
        capture.observe(self.empty, False, 3.5)
        self.assertTrue(capture.samples)
        self.assertIsNone(capture.observe(self.person, True, 3.9))
        self.assertFalse(capture.samples)
        self.assertIsNone(capture.observe(self.empty, False, 4))
        self.assertIsNone(capture.observe(self.empty, False, 7.9))
        np.testing.assert_array_equal(capture.observe(self.empty, False, 8), self.empty)

    def test_reset_requires_another_clean_capture(self):
        capture = CleanBackgroundCapture()
        capture.observe(self.empty, False, 0)
        capture.observe(self.empty, False, 4)
        capture.reset()
        self.assertEqual(capture.status, "waiting")
        self.assertIsNone(capture.observe(self.empty, False, 10))
        self.assertIsNone(capture.observe(self.person, True, 14))

    def test_preview_message_does_not_modify_video_frame(self):
        frame = np.full((540, 960, 3), 80, np.uint8)
        original = frame.copy()
        for status in ("waiting", "capturing", "ready"):
            preview = calibration_preview(frame, status, 3)
            self.assertEqual(preview.shape, frame.shape)
            self.assertFalse(np.array_equal(preview, frame))
            np.testing.assert_array_equal(frame, original)


if __name__ == "__main__":
    unittest.main()
