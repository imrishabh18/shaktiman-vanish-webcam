"""Turn the captured person into a body-colored whirlwind that flies upward.

The person's horizontal color bands wrap around a tapered, twisting volume.
Premultiplied alpha keeps both deformation and motion blur free of dark seams.
The room is never warped. No show footage or external visual assets are used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


SHAKTIMAAN_SECONDS = 2.4
_TAU = math.tau


@dataclass(frozen=True)
class ShaktimaanSnapshot:
    pixels: np.ndarray  # Cropped float32 premultiplied BGR plus alpha.
    left: int
    top: int
    frame_shape: tuple[int, int]
    axis_x: float
    surface: np.ndarray  # Periodic texture made only from foreground colors.
    radius: np.ndarray
    coverage: np.ndarray

    @property
    def center(self) -> tuple[float, float]:
        return self.left + self.axis_x, self.top + self.pixels.shape[0] * 0.5


def _smooth(start: float, end: float, value: float) -> float:
    t = min(1.0, max(0.0, (value - start) / (end - start)))
    return t * t * (3.0 - 2.0 * t)


def _surface_from_person(pixels: np.ndarray):
    """Unwrap the body into horizontal strips without sampling room pixels."""
    height, width = pixels.shape[:2]
    alpha = pixels[:, :, 3]
    mass = alpha.sum(axis=0)
    axis_x = float(np.dot(np.arange(width), mass) / max(float(mass.sum()), 1e-6))
    strips = np.zeros((height, 256, 3), np.float32)
    radius = np.zeros(height, np.float32)
    coverage = np.zeros(height, np.float32)
    populated = []
    for y in range(height):
        visible = np.flatnonzero(alpha[y] > 0.15)
        if visible.size == 0:
            continue
        valid = np.flatnonzero(alpha[y] > 0.65)
        if valid.size == 0:
            valid = visible
        colors = pixels[y, valid, :3] / np.maximum(alpha[y, valid, None], 1e-6)
        positions = np.linspace(0, valid.size - 1, 256)
        for channel in range(3):
            strips[y, :, channel] = np.interp(positions, np.arange(valid.size), colors[:, channel])
        radius[y] = max(1.0, (visible[-1] - visible[0] + 1) * 0.59)
        coverage[y] = min(1.0, float(alpha[y, valid].max()) * 1.2)
        populated.append(y)
    # Fill only the texture in missing rows. Coverage still controls visibility.
    if populated:
        rows = np.asarray(populated)
        missing = np.flatnonzero(coverage == 0)
        if missing.size:
            nearest = np.abs(missing[:, None] - rows[None, :]).argmin(axis=1)
            strips[missing] = strips[rows[nearest]]
    sigma = max(0.8, height * 0.028)
    radius = cv2.GaussianBlur(radius[:, None], (1, 0), sigmaX=0, sigmaY=sigma)[:, 0]
    coverage = cv2.GaussianBlur(coverage[:, None], (1, 0), sigmaX=0, sigmaY=max(0.7, height * 0.006))[:, 0]
    # Join the feet into a tapered spinning tip rather than a wide, flat base.
    bottom = np.clip(np.linspace(1, 0, height, dtype=np.float32) / 0.16, 0, 1)
    radius *= bottom * bottom * (3 - 2 * bottom)
    # Front and mirrored back make a seamless revolution. Blur in the direction
    # of rotation, retaining the actual clothing/skin bands along the body.
    surface = np.concatenate((strips, strips[:, ::-1]), axis=1)
    smeared = cv2.GaussianBlur(surface, (0, 1), sigmaX=78, sigmaY=0)
    surface = surface * 0.025 + smeared * 0.975
    return axis_x, surface, radius, coverage


def capture_shaktimaan(frame: np.ndarray, mask: np.ndarray) -> ShaktimaanSnapshot | None:
    """Capture the person and prepare the whirlwind texture once per trigger."""
    if frame.ndim != 3 or frame.shape[2] != 3 or mask.shape != frame.shape[:2]:
        raise ValueError("The BGR frame and person mask must have matching dimensions.")
    alpha = np.clip(np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0), 0, 1).astype(np.float32)
    visible = (alpha > 0.15).astype(np.uint8)
    if np.count_nonzero(visible) < max(24, mask.size * 0.0004):
        return None
    x, y, width, height = cv2.boundingRect(cv2.findNonZero(visible))
    if width < 4 or height < 8:
        return None
    left, top = max(0, x - 3), max(0, y - 3)
    right, bottom = min(frame.shape[1], x + width + 3), min(frame.shape[0], y + height + 3)
    crop_alpha = alpha[top:bottom, left:right, None]
    pixels = np.concatenate((frame[top:bottom, left:right].astype(np.float32) * crop_alpha, crop_alpha), axis=2)
    axis, surface, radius, coverage = _surface_from_person(pixels)
    return ShaktimaanSnapshot(pixels, left, top, frame.shape[:2], axis, surface, radius, coverage)


def _over(canvas: np.ndarray, pixels: np.ndarray, left: int, top: int, opacity: float = 1.0) -> None:
    height, width = pixels.shape[:2]
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(canvas.shape[1], left + width), min(canvas.shape[0], top + height)
    if x0 >= x1 or y0 >= y1 or opacity <= 0:
        return
    part = pixels[y0 - top:y1 - top, x0 - left:x1 - left]
    alpha = part[:, :, 3:4] * opacity
    destination = canvas[y0:y1, x0:x1]
    destination[:] = np.clip(part[:, :, :3] * opacity + destination * (1 - alpha), 0, 255)


def _whirlwind_layer(snapshot: ShaktimaanSnapshot, age: float):
    """Twist row by row into an opaque, cylindrical body with soft moving edges."""
    height, width = snapshot.pixels.shape[:2]
    p = age / SHAKTIMAAN_SECONDS
    morph = _smooth(0.035, 0.29, p)
    half_width = max(snapshot.axis_x, width - snapshot.axis_x, float(snapshot.radius.max())) * 1.25 + 8
    # Motion blur permits a smaller working surface, keeping 720p output fast.
    quality = min(1.0, 460 / height, 600 / (2 * half_width))
    rows = max(8, round(height * quality))
    columns = max(8, round(2 * half_width * quality))
    source_y = np.linspace(0, height - 1, rows, dtype=np.float32)[:, None]
    v = source_y / max(height - 1, 1)
    x = np.linspace(-half_width, half_width, columns, dtype=np.float32)[None, :]
    turns = _smooth(0.01, 0.28, p) * 2.8
    phase = (_TAU * (1.4 * age + 3.8 * age * age) + (0.5 - v) * _TAU * turns).astype(np.float32)
    radius = np.interp(source_y[:, 0], np.arange(height), snapshot.radius).astype(np.float32)[:, None]
    radius = np.maximum(1.0, radius * (1.02 + 0.04 * np.sin(phase)))
    wobble = np.sin(phase) * np.minimum(radius * 0.06, height * 0.012) * morph
    u = (x - wobble) / radius
    angle = np.arcsin(np.clip(u, -0.999, 0.999))
    map_x = (((angle + np.pi * 0.5 + phase) / _TAU) % 1.0 * (snapshot.surface.shape[1] - 1)).astype(np.float32)
    map_y = np.broadcast_to(source_y, (rows, columns)).copy()
    map_y += np.sin(phase + angle * 1.1) * height * 0.009 * morph
    colors = cv2.remap(snapshot.surface, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # Volume shading and a soft contour make a blurred figure, not bright rings.
    front = np.sqrt(np.maximum(0, 1 - u * u))
    shading = (0.77 + front * 0.23) * (1 + 0.045 * np.sin(phase + angle * 2))
    # Fine streaks are variations in the existing surface brightness. They
    # follow the spinning material instead of drawing a separate spiral cage.
    shading *= 0.97 + 0.03 * np.sin(v * _TAU * 33 + phase * 0.6 + angle * 1.5)
    colors *= shading[:, :, None]
    edge = np.clip((1 - np.abs(u)) * np.maximum(radius * quality, 1.0), 0, 1)
    coverage = np.interp(source_y[:, 0], np.arange(height), snapshot.coverage).astype(np.float32)[:, None]
    alpha = edge * coverage
    volume = np.concatenate((colors * alpha[:, :, None], alpha[:, :, None]), axis=2)

    if morph < 0.999:
        facing = np.cos(phase)
        facing = np.where(facing >= 0, 1, -1) * np.maximum(np.abs(facing), 0.19)
        body_x = (snapshot.axis_x + (x - wobble) / facing).astype(np.float32)
        body_y = np.broadcast_to(source_y, body_x.shape).copy()
        body_y += np.sin(phase) * height * 0.008 * morph
        body = cv2.remap(snapshot.pixels, body_x, body_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if morph > 0.02:
            body = cv2.GaussianBlur(body, (0, 1), sigmaX=max(0.2, columns * 0.018 * morph), sigmaY=0)
        volume = body * (1 - morph) + volume * morph

    # A directional smear made from these same pixels reinforces the spin.
    soft = cv2.GaussianBlur(volume, (0, 1), sigmaX=max(0.3, columns * 0.032), sigmaY=0)
    volume = volume * (1 - 0.52 * morph) + soft * (0.52 * morph)
    return volume, half_width


def render_shaktimaan(background: np.ndarray, snapshot: ShaktimaanSnapshot | None, age: float) -> np.ndarray:
    """Body twist, clothing-colored whirlwind, then accelerating upward flight."""
    if snapshot is None or age >= SHAKTIMAAN_SECONDS or not math.isfinite(age):
        return background.copy()
    if background.shape[:2] != snapshot.frame_shape:
        raise ValueError("The captured person and background dimensions must match.")
    output = background.copy()
    if age <= 0:
        _over(output, snapshot.pixels, snapshot.left, snapshot.top)
        return output

    p = age / SHAKTIMAAN_SECONDS
    launch = float(np.clip((p - 0.53) / 0.43, 0, 1))
    height = snapshot.pixels.shape[0]
    # Preserve almost the full body height: leave through the top of the frame
    # instead of collapsing into a tiny point or dissolving in place.
    travel = snapshot.top + height * 1.24
    lift = travel * launch * launch
    scale = 1 - 0.035 * launch
    top = round(snapshot.top - lift)
    target_height = max(1, round(height * scale))
    velocity = travel * 2 * launch / (SHAKTIMAAN_SECONDS * 0.43)
    streak = min(height * 0.16, velocity * 0.024)
    if top + target_height + streak < 0:
        return output

    layer, half_width = _whirlwind_layer(snapshot, age)
    target_width = max(1, round(2 * half_width * scale))
    layer = cv2.resize(layer, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    left = round(snapshot.center[0] - target_width * 0.5)
    if streak > 1:
        for offset, opacity in ((1.0, 0.045), (0.66, 0.07), (0.33, 0.10)):
            _over(output, layer, left, top + round(streak * offset), opacity)
    _over(output, layer, left, top)
    return output
