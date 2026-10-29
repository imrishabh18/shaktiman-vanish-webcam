"""Capture a clean room only after the person has left the camera view."""

from dataclasses import dataclass, field
import math

import cv2
import numpy as np


def mask_has_person(mask: np.ndarray) -> bool:
    """Reject a substantial connected person region, allowing tiny mask noise."""
    if mask.ndim != 2 or not mask.size or not np.isfinite(mask).all():
        return True
    height, width = mask.shape
    scale = min(1.0, 320 / width)
    small = cv2.resize(mask, (max(1, round(width * scale)), max(1, round(height * scale))))
    binary = (small > 0.55).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    for x, y, region_width, region_height, area in stats[1:]:
        if area < max(12, binary.size * 0.0035) or region_height < binary.shape[0] * 0.18:
            continue
        # The landscape model can mark a narrow door-frame edge as foreground,
        # or a short strip of bedding at the bottom. Neither is a person shape.
        edge_strip = (
            region_width <= binary.shape[1] * 0.065
            and region_height >= binary.shape[0] * 0.8
            and (x <= 1 or x + region_width >= binary.shape[1] - 1)
        )
        if not edge_strip:
            return True
    return False


@dataclass
class CleanBackgroundCapture:
    seconds: float = 4.0
    sample_seconds: float = 0.55
    started: float | None = None
    status: str = "waiting"
    remaining: int = 4
    samples: list[np.ndarray] = field(default_factory=list)

    def reset(self) -> None:
        self.started = None
        self.status = "waiting"
        self.remaining = math.ceil(self.seconds)
        self.samples.clear()

    def observe(self, frame: np.ndarray, person_present: bool, now: float) -> np.ndarray | None:
        """Only accept frames that have just passed the person check."""
        if person_present or not math.isfinite(now):
            self.reset()
            return None
        if self.started is None or now < self.started:
            self.started = now
            self.samples.clear()
        elapsed = now - self.started
        self.status = "capturing"
        self.remaining = max(0, math.ceil(self.seconds - elapsed - 1e-6))
        if elapsed >= self.seconds - self.sample_seconds:
            self.samples.append(frame.copy())
            self.samples[:] = self.samples[-18:]
        if elapsed + 1e-6 < self.seconds:
            return None
        # Include the current checked frame, even at a very low camera FPS.
        if not self.samples:
            self.samples.append(frame.copy())
        background = np.median(np.stack(self.samples), axis=0).astype(np.uint8)
        self.samples.clear()
        self.status = "ready"
        return background


def calibration_preview(frame: np.ndarray, status: str, remaining: int) -> np.ndarray:
    """Setup instructions on the local preview only, never on virtual output."""
    output = frame.copy()
    height, width = output.shape[:2]
    bar = min(height, max(64, round(height * 0.16)))
    region = output[:bar]
    region[:] = (region.astype(np.float32) * 0.18).astype(np.uint8)
    if status == "ready":
        heading = "BACKGROUND READY - STEP BACK IN"
        detail = "You can return to view. V to test. R to reappear."
        color = (120, 245, 160)
    elif status == "capturing":
        heading = f"CAPTURING EMPTY ROOM - {remaining}s"
        detail = "Stay out of view until BACKGROUND READY appears."
        color = (110, 220, 255)
    else:
        heading = "STEP OUT OF THE CAMERA VIEW"
        detail = "Waiting for an empty room. Move fully out of view."
        color = (110, 220, 255)
    scale = min(0.85, max(0.25, width / 1200))
    x = max(4, round(width * 0.022))
    y = min(bar - 20, max(17, round(bar * 0.43)))
    cv2.putText(output, heading, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    cv2.putText(output, detail, (x, min(bar - 5, y + max(14, round(28 * scale)))),
                cv2.FONT_HERSHEY_SIMPLEX, scale * 0.65, (235, 235, 235), 1, cv2.LINE_AA)
    return output
