"""Image geometry and astronomical position-angle conventions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import affine_transform


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def circular_distance_deg(a: float | np.ndarray, b: float | np.ndarray):
    """Return the unsigned shortest separation between position angles."""
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


def position_angle_deg(
    x: float | np.ndarray,
    y: float | np.ndarray,
    *,
    center_x: float,
    center_y: float,
):
    """Return PA with north=0, east=90 and counterclockwise in north-up data.

    ``x`` grows to image right and ``y`` grows down the array, so astronomical
    east is image left in a north-up image.
    """
    dx_east = center_x - np.asarray(x)
    dy_north = center_y - np.asarray(y)
    return np.degrees(np.arctan2(dx_east, dy_north)) % 360.0


@dataclass(frozen=True)
class Crop:
    """Explicit half-open pixel bounds ``left, top, right, bottom``."""

    left: int
    top: int
    right: int
    bottom: int

    def apply(self, image: np.ndarray) -> np.ndarray:
        if not (0 <= self.left < self.right <= image.shape[1]):
            raise ValueError("crop horizontal bounds fall outside the image")
        if not (0 <= self.top < self.bottom <= image.shape[0]):
            raise ValueError("crop vertical bounds fall outside the image")
        return image[self.top : self.bottom, self.left : self.right].copy()


@dataclass(frozen=True)
class Alignment:
    """Documented similarity transform from a source image to north-up.

    Positive ``rotation_deg`` rotates the visible source image counterclockwise
    in array display coordinates (columns right, rows down). Thus +90 degrees
    moves a source marker on image-right to output-up. The affine matrix maps
    output coordinates back to source coordinates, as SciPy requires.
    """

    center_x_px: float
    center_y_px: float
    solar_radius_px: float
    rotation_deg: float
    reflected: bool = False
    output_size_px: int = 789
    output_solar_radius_px: float = 789.0 / 5.6

    def __post_init__(self) -> None:
        for field in (
            "center_x_px", "center_y_px", "solar_radius_px", "rotation_deg",
            "output_solar_radius_px",
        ):
            object.__setattr__(self, field, _finite_number(getattr(self, field), field))
        if isinstance(self.output_size_px, bool) or not isinstance(self.output_size_px, int):
            raise ValueError("output_size_px must be a positive integer")
        if self.solar_radius_px <= 0 or self.output_solar_radius_px <= 0:
            raise ValueError("solar radii must be positive")
        if self.output_size_px <= 0:
            raise ValueError("output size must be positive")
        if not -180.0 <= self.rotation_deg <= 180.0:
            raise ValueError("rotation_deg must be within [-180, 180]")
        if self.output_solar_radius_px > self.output_size_px / 2.0:
            raise ValueError("output solar radius must fit inside the output image")


def align_north_up(
    image: np.ndarray,
    alignment: Alignment,
    *,
    order: int = 1,
    cval: float = np.nan,
) -> np.ndarray:
    """Apply only the manifest-declared center, scale, rotation and parity.

    A positive rotation is a counterclockwise correction of the displayed
    source image: +90 degrees moves source-right to output-up. Rotation is fixed
    by external geometry and is never estimated by image correlation.
    """
    theta = np.radians(alignment.rotation_deg)
    parity = -1.0 if alignment.reflected else 1.0
    scale = alignment.solar_radius_px / alignment.output_solar_radius_px

    # Coordinates here are (row, column). This maps output pixels back into the
    # source in one interpolation, avoiding chained rotate/zoom operations.
    matrix = scale * np.array(
        [[np.cos(theta), parity * np.sin(theta)],
         [-np.sin(theta), parity * np.cos(theta)]],
        dtype=float,
    )
    out_center = np.full(2, (alignment.output_size_px - 1) / 2.0)
    source_center = np.array([alignment.center_y_px, alignment.center_x_px])
    offset = source_center - matrix @ out_center

    channels = [image] if image.ndim == 2 else np.moveaxis(image, -1, 0)
    transformed = [
        affine_transform(
            channel,
            matrix,
            offset=offset,
            output_shape=(alignment.output_size_px,) * 2,
            order=order,
            mode="constant",
            cval=cval,
            prefilter=order > 1,
        )
        for channel in channels
    ]
    return transformed[0] if image.ndim == 2 else np.moveaxis(transformed, 0, -1)
