import math
import unittest
from types import SimpleNamespace

from shaktiman.gesture import PointUpTrigger, pointing_up_score


def hand_points(extended=(True, False, False, False)):
    """Pixel-space pose in a square frame, with optional extended fingers."""
    points = [(320.0, 420.0)] * 21
    for base, x, opened in zip((5, 9, 13, 17), (275, 315, 350, 380), extended):
        points[base:base + 4] = (
            [(x, 300), (x, 235), (x, 185), (x, 145)] if opened
            else [(x, 300), (x, 260), (x, 300), (x, 340)]
        )
    return points


def landmarks(points, width=640, height=480):
    return [SimpleNamespace(x=x / width, y=y / height, z=0.0) for x, y in points]


class PointingUpTests(unittest.TestCase):
    def test_mirror_scale_translation_and_moderate_tilt(self):
        for mirror in (-1, 1):
            for scale in (0.3, 1.0, 2.0):
                for angle in (-0.45, 0, 0.45):
                    for width, height in ((640, 480), (1280, 720), (480, 960)):
                        c, s = math.cos(angle), math.sin(angle)
                        transformed = []
                        for x, y in hand_points():
                            x, y = (x - 320) * mirror * scale, (y - 420) * scale
                            transformed.append((x * c - y * s + 600, x * s + y * c + 700))
                        with self.subTest(mirror=mirror, scale=scale, angle=angle, size=(width, height)):
                            self.assertGreater(pointing_up_score(landmarks(transformed, width, height), width, height), 0.88)

    def test_thumb_is_ignored(self):
        for thumb in (((220, 350),) * 4, ((180, 300), (160, 260), (140, 220), (120, 180))):
            points = hand_points()
            points[1:5] = thumb
            self.assertGreater(pointing_up_score(landmarks(points), 640, 480), 0.88)

    def test_horizontal_and_downward_points_are_rejected(self):
        for angle in (math.pi / 2, -math.pi / 2, math.pi):
            c, s = math.cos(angle), math.sin(angle)
            points = [(x * c - y * s, x * s + y * c) for x, y in hand_points()]
            self.assertEqual(pointing_up_score(landmarks(points), 640, 480), 0.0)

    def test_other_poses_are_rejected(self):
        for extended in ((False,) * 4, (True,) * 4, (True, True, False, False), (True, False, False, True)):
            with self.subTest(extended=extended):
                self.assertEqual(pointing_up_score(landmarks(hand_points(extended)), 640, 480), 0.0)

    def test_bent_index_is_rejected(self):
        points = hand_points()
        points[8] = (275, 280)
        self.assertEqual(pointing_up_score(landmarks(points), 640, 480), 0.0)

    def test_invalid_inputs_are_rejected(self):
        valid = landmarks(hand_points())
        for bad in (None, [], valid[:20], [object()] * 21, landmarks([(0, 0)] * 21)):
            with self.subTest(bad=bad):
                self.assertEqual(pointing_up_score(bad, 640, 480), 0.0)
        for value in (math.nan, math.inf, -math.inf):
            bad = landmarks(hand_points())
            bad[8].x = value
            self.assertEqual(pointing_up_score(bad, 640, 480), 0.0)
        for width, height in ((0, 480), (640, -1), (math.nan, 480), (640, math.inf)):
            self.assertEqual(pointing_up_score(valid, width, height), 0.0)


class PointUpTriggerTests(unittest.TestCase):
    def test_hold_boundary_and_held_finger_cannot_repeat(self):
        trigger = PointUpTrigger()
        self.assertFalse(trigger.update(True, 10.0))
        self.assertFalse(trigger.update(True, 10.14))
        self.assertFalse(trigger.update(True, 10.28))
        self.assertFalse(trigger.update(True, 10.349))
        self.assertTrue(trigger.update(True, 10.35))
        self.assertFalse(trigger.update(True, 12.0))
        self.assertFalse(trigger.update(True, 13.0))

    def test_release_resets_pending_hold_and_rearms_after_success(self):
        trigger = PointUpTrigger()
        self.assertFalse(trigger.update(True, 0))
        self.assertFalse(trigger.update(False, 0.3))
        self.assertFalse(trigger.update(True, 0.4))
        self.assertFalse(trigger.update(True, 0.55))
        self.assertFalse(trigger.update(True, 0.7))
        self.assertTrue(trigger.update(True, 0.75))
        self.assertFalse(trigger.update(False, 1.0))
        self.assertFalse(trigger.update(True, 1.1))
        self.assertFalse(trigger.update(True, 1.3))
        self.assertTrue(trigger.update(True, 1.45))

    def test_return_reset_requires_release_before_new_hold(self):
        trigger = PointUpTrigger()
        trigger.reset(require_release=True)
        self.assertFalse(trigger.update(True, 1.0))
        self.assertFalse(trigger.update(True, 2.0))
        self.assertFalse(trigger.update(False, 2.1))
        self.assertFalse(trigger.update(True, 2.2))
        self.assertFalse(trigger.update(True, 2.4))
        self.assertTrue(trigger.update(True, 2.55))

    def test_not_ready_cannot_accumulate_a_hold(self):
        trigger = PointUpTrigger()
        self.assertFalse(trigger.update(True, 1.0))
        self.assertFalse(trigger.update(True, 1.5, ready=False))
        self.assertFalse(trigger.update(True, 2.0))
        self.assertFalse(trigger.update(True, 2.2))
        self.assertFalse(trigger.update(True, 2.3))
        self.assertTrue(trigger.update(True, 2.35))

    def test_reset_and_bad_time_clear_pending_hold(self):
        trigger = PointUpTrigger()
        self.assertFalse(trigger.update(True, 1.0))
        trigger.reset()
        self.assertFalse(trigger.update(True, 2.0))
        self.assertFalse(trigger.update(True, math.nan))
        self.assertFalse(trigger.update(True, 3.0))
        self.assertFalse(trigger.update(True, 2.0))
        self.assertFalse(trigger.update(True, 2.2))
        self.assertTrue(trigger.update(True, 2.35))

    def test_a_long_observation_gap_starts_a_fresh_hold(self):
        trigger = PointUpTrigger()
        self.assertFalse(trigger.update(True, 0.0))
        self.assertFalse(trigger.update(True, 0.2))
        self.assertFalse(trigger.update(True, 1.0))
        self.assertFalse(trigger.update(True, 1.2))
        self.assertTrue(trigger.update(True, 1.35))


if __name__ == "__main__":
    unittest.main()
