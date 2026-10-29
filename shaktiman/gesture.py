"""Screen-up index-finger recognition and a deliberate, one-shot trigger.

Use landmarks from the original, unrotated image: unlike a generic pointing
gesture, this pose must face toward the top of the camera frame.
"""

from __future__ import annotations

import math


def _distance(first, second):
    return math.hypot(first[0] - second[0], first[1] - second[1])


def pointing_up_score(landmarks, width: int, height: int) -> float:
    """Return index straightness for an upward point, or zero for other poses.

    MediaPipe's normalized coordinates are converted to pixels before comparing
    directions, so a widescreen image does not change the permitted hand tilt.
    The thumb may be open or closed. Both left and right hands are accepted.
    """
    try:
        if len(landmarks) != 21:
            return 0.0
        width, height = float(width), float(height)
        if not (math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0):
            return 0.0
        points = [(float(point.x) * width, float(point.y) * height) for point in landmarks]
    except (AttributeError, TypeError, ValueError, OverflowError):
        return 0.0
    if not all(math.isfinite(value) for point in points for value in point):
        return 0.0

    palm_length = _distance(points[0], points[9])
    if palm_length <= 1e-6:
        return 0.0

    fingers = []
    for base in (5, 9, 13, 17):
        finger = points[base:base + 4]
        length = sum(_distance(first, second) for first, second in zip(finger, finger[1:]))
        chord = _distance(finger[0], finger[3])
        straightness = chord / length if length > 1e-6 else 0.0
        fingers.append((length, chord, straightness))

    index_length, index_chord, straightness = fingers[0]
    if index_length < 0.35 * palm_length or index_chord <= 1e-6 or straightness < 0.88:
        return 0.0
    # A moderate sideways tilt is allowed, but horizontal/downward points fail.
    upward = (points[5][1] - points[8][1]) / index_chord
    if upward < 0.75:
        return 0.0
    if not (
        points[6][1] < points[5][1] - 0.12 * index_length
        and points[8][1] < points[6][1] - 0.18 * index_length
        and _distance(points[8], points[0]) > 1.10 * _distance(points[6], points[0])
    ):
        return 0.0

    index_reach = _distance(points[8], points[0])
    for base, (length, chord, ratio) in zip((9, 13, 17), fingers[1:]):
        tip_reach = _distance(points[base + 3], points[0])
        joint_reach = _distance(points[base + 1], points[0])
        folded = (
            tip_reach <= 1.05 * joint_reach
            or (ratio < 0.78 and tip_reach < index_reach * 0.88)
            # Occluded curled fingers can project to short, straight segments.
            or (chord < index_length * 0.48 and tip_reach < index_reach * 0.85)
        )
        if not folded:
            return 0.0
    return min(1.0, straightness)


class PointUpTrigger:
    """Require a continuous hold and an observed release before the next trigger."""

    def __init__(self, hold_seconds: float = 0.35, max_gap_seconds: float = 0.25):
        if not math.isfinite(hold_seconds) or hold_seconds <= 0:
            raise ValueError("hold_seconds must be positive and finite")
        if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be positive and finite")
        self.hold_seconds = hold_seconds
        self.max_gap_seconds = max_gap_seconds
        self.reset()

    def reset(self, require_release: bool = False) -> None:
        """Clear a pending hold; R should require a fresh release and point."""
        self._started = None
        self._last_seen = None
        self._armed = not require_release

    def update(self, pointing: bool, now: float, ready: bool = True) -> bool:
        if not math.isfinite(now):
            self._started = None
            self._last_seen = None
            return False
        if not pointing:
            self._started = None
            self._last_seen = None
            self._armed = True
            return False
        if not ready:
            self._started = None
            self._last_seen = None
            return False
        if not self._armed:
            return False
        if (
            self._started is None
            or now < self._last_seen
            or now - self._last_seen > self.max_gap_seconds + 1e-9
        ):
            self._started = now
            self._last_seen = now
            return False
        self._last_seen = now
        if now - self._started + 1e-9 < self.hold_seconds:
            return False
        self._started = None
        self._last_seen = None
        self._armed = False
        return True
