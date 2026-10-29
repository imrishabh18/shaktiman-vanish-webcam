import unittest

import cv2
import numpy as np

from shaktiman.effect import (
    SHAKTIMAAN_SECONDS, capture_shaktimaan, render_shaktimaan,
)


class ShaktimaanEffectTests(unittest.TestCase):
    def setUp(self):
        self.background = np.full((180, 260, 3), (70, 45, 20), dtype=np.uint8)
        self.mask = np.zeros((180, 260), dtype=np.float32)
        cv2.ellipse(self.mask, (130, 100), (30, 65), 0, 0, 360, 1, -1)
        self.frame = self.background.copy()
        self.frame[self.mask > 0] = (20, 80, 210)
        self.snapshot = capture_shaktimaan(self.frame, self.mask)

    def test_initial_frame_preserves_captured_person_and_stationary_background(self):
        np.testing.assert_array_equal(
            render_shaktimaan(self.background, self.snapshot, 0), self.frame,
        )

    def test_animation_changes_then_ends_on_exact_background(self):
        middle = render_shaktimaan(self.background, self.snapshot, 0.9)
        self.assertGreater(np.count_nonzero(middle != self.frame), 100)
        self.assertGreater(np.count_nonzero(middle != self.background), 100)
        for age in (SHAKTIMAAN_SECONDS, SHAKTIMAAN_SECONDS + 1):
            np.testing.assert_array_equal(
                render_shaktimaan(self.background, self.snapshot, age), self.background,
            )

    def test_capture_is_frozen_and_does_not_modify_inputs(self):
        original = self.frame.copy()
        expected = render_shaktimaan(self.background, self.snapshot, 0.5)
        self.frame[:] = 0
        self.mask[:] = 0
        np.testing.assert_array_equal(
            render_shaktimaan(self.background, self.snapshot, 0.5), expected,
        )
        np.testing.assert_array_equal(self.background, np.full_like(original, (70, 45, 20)))

    def test_empty_tiny_and_invalid_masks_do_not_create_a_person(self):
        for mask in (np.zeros_like(self.mask), np.full_like(self.mask, np.nan)):
            self.assertIsNone(capture_shaktimaan(self.frame, mask))
        tiny = np.zeros_like(self.mask)
        tiny[10:12, 10:12] = 1
        self.assertIsNone(capture_shaktimaan(self.frame, tiny))
        np.testing.assert_array_equal(
            render_shaktimaan(self.background, None, 0.8), self.background,
        )

    def test_soft_mask_uses_premultiplied_color(self):
        frame = np.full((32, 32, 3), (100, 200, 240), dtype=np.uint8)
        mask = np.full((32, 32), 0.5, dtype=np.float32)
        background = np.full_like(frame, (20, 40, 60))
        snapshot = capture_shaktimaan(frame, mask)
        expected = np.full_like(frame, (60, 120, 150))
        np.testing.assert_array_equal(render_shaktimaan(background, snapshot, 0), expected)

    def test_person_at_each_frame_edge_stays_valid_through_animation(self):
        for position in ((0, 0), (260, 0), (0, 180), (260, 180)):
            mask = np.zeros_like(self.mask)
            cv2.ellipse(mask, position, (55, 110), 0, 0, 360, 1, -1)
            snapshot = capture_shaktimaan(self.frame, mask)
            self.assertIsNotNone(snapshot)
            for age in np.linspace(0, SHAKTIMAAN_SECONDS, 15):
                with self.subTest(position=position, age=age):
                    image = render_shaktimaan(self.background, snapshot, float(age))
                    self.assertEqual(image.shape, self.frame.shape)
                    self.assertEqual(image.dtype, np.uint8)

    def test_dimension_mismatch_is_reported(self):
        with self.assertRaises(ValueError):
            capture_shaktimaan(self.frame, self.mask[:10])
        with self.assertRaises(ValueError):
            render_shaktimaan(self.background[:10], self.snapshot, 0.5)

    def test_vortex_keeps_person_colors_without_painted_energy_rings(self):
        background = np.full((300, 300, 3), (0, 90, 0), np.uint8)
        mask = np.zeros((300, 300), np.float32)
        cv2.ellipse(mask, (150, 160), (45, 120), 0, 0, 360, 1, -1)
        frame = background.copy()
        frame[mask > 0] = (180, 0, 0)
        snapshot = capture_shaktimaan(frame, mask)
        for age in (0.6, 1.0, 1.7):
            result = render_shaktimaan(background, snapshot, age)
            self.assertTrue(np.all(result[:, :, 2] == 0))
            self.assertGreater(np.count_nonzero(result[:, :, 0]), 100)

    def test_whirlwind_flies_up_at_nearly_full_height(self):
        background = np.full((900, 400, 3), 30, np.uint8)
        mask = np.zeros((900, 400), np.float32)
        cv2.ellipse(mask, (200, 570), (50, 140), 0, 0, 360, 1, -1)
        frame = background.copy()
        frame[mask > 0] = (80, 150, 210)
        snapshot = capture_shaktimaan(frame, mask)
        bounds = []
        for age in (1.1, 1.8):
            output = render_shaktimaan(background, snapshot, age)
            changed = (np.max(np.abs(output.astype(float) - background), axis=2) > 4).astype(np.uint8)
            bounds.append(cv2.boundingRect(cv2.findNonZero(changed)))
        self.assertGreater(bounds[1][3], 0.8 * bounds[0][3])
        self.assertLess(bounds[1][1] + bounds[1][3], bounds[0][1] + bounds[0][3] - 80)

    def test_room_stays_pixel_identical_outside_the_body(self):
        rng = np.random.default_rng(16)
        background = rng.integers(0, 255, (300, 600, 3), dtype=np.uint8)
        mask = np.zeros((300, 600), np.float32)
        mask[20:280, 270:330] = 1
        frame = background.copy()
        frame[mask > 0] = (30, 90, 170)
        snapshot = capture_shaktimaan(frame, mask)
        for age in (0.2, 0.8, 1.6, 2.1):
            output = render_shaktimaan(background, snapshot, age)
            np.testing.assert_array_equal(output[:, :150], background[:, :150])
            np.testing.assert_array_equal(output[:, 450:], background[:, 450:])


if __name__ == "__main__":
    unittest.main()
