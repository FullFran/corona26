"""Angular profile sampling and circular streamer-peak detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.signal import find_peaks


@dataclass(frozen=True)
class AngularProfile:
    radius_rsun: float
    pa_deg: np.ndarray
    values: np.ndarray
    coverage: float
    median: float | None
    mad: float | None
    status: str


def robust_normalize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Normalize finite samples by median and raw median absolute deviation."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 3:
        raise ValueError("not enough finite profile samples")
    median = float(np.median(values[finite]))
    mad = float(np.median(np.abs(values[finite] - median)))
    if not np.isfinite(mad) or mad <= 0:
        raise ValueError("profile has zero or invalid MAD")
    return (values - median) / mad, median, mad


def _intensity(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[..., :3].mean(axis=2)
    raise ValueError("image must be a 2-D intensity or RGB(A) array")


def sample_angular_profile(
    image: np.ndarray,
    *,
    center_x_px: float,
    center_y_px: float,
    solar_radius_px: float,
    radius_rsun: float,
    mask: np.ndarray | None = None,
    half_width_rsun: float = 0.05,
    pa_step_deg: float = 1.0,
    radial_samples: int = 11,
    minimum_coverage: float = 0.8,
) -> AngularProfile:
    """Sample median intensity across a narrow annulus at fixed PA bins."""
    if half_width_rsun <= 0 or radial_samples < 2 or pa_step_deg <= 0:
        raise ValueError("invalid profile sampling parameters")
    intensity = _intensity(image)
    if mask is None:
        mask_array = np.isfinite(intensity)
    else:
        if mask.shape != intensity.shape:
            raise ValueError("mask shape must match the intensity image")
        mask_array = np.asarray(mask, dtype=bool) & np.isfinite(intensity)

    pa = np.arange(0.0, 360.0, pa_step_deg)
    radii = np.linspace(
        radius_rsun - half_width_rsun,
        radius_rsun + half_width_rsun,
        radial_samples,
    ) * solar_radius_px
    angle = np.radians(pa)[:, None]
    x = center_x_px - np.sin(angle) * radii[None, :]
    y = center_y_px - np.cos(angle) * radii[None, :]
    coords = np.vstack((y.ravel(), x.ravel()))
    samples = map_coordinates(intensity, coords, order=1, mode="constant", cval=np.nan)
    valid = map_coordinates(
        mask_array.astype(float), coords, order=0, mode="constant", cval=0.0
    ).reshape(pa.size, radial_samples) > 0.5
    samples = samples.reshape(pa.size, radial_samples)
    valid &= np.isfinite(samples)
    coverage = float(valid.mean())

    values = np.full(pa.size, np.nan)
    for index in range(pa.size):
        if valid[index].any():
            values[index] = np.median(samples[index, valid[index]])

    if coverage < minimum_coverage:
        return AngularProfile(radius_rsun, pa, values, coverage, None, None, "not_enough_coverage")
    try:
        normalized, median, mad = robust_normalize(values)
    except ValueError:
        return AngularProfile(radius_rsun, pa, values, coverage, None, None, "degenerate")
    return AngularProfile(radius_rsun, pa, normalized, coverage, median, mad, "ok")


def smooth_circular_profile(
    values: np.ndarray,
    *,
    sigma_deg: float = 3.0,
    pa_step_deg: float = 1.0,
) -> np.ndarray:
    """Gaussian-smooth a possibly masked profile without breaking the seam."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return np.full_like(values, np.nan)
    sigma = sigma_deg / pa_step_deg
    numerator = gaussian_filter1d(np.where(finite, values, 0.0), sigma, mode="wrap")
    weight = gaussian_filter1d(finite.astype(float), sigma, mode="wrap")
    return np.divide(
        numerator,
        weight,
        out=np.full_like(numerator, np.nan),
        where=weight > 1e-6,
    )


def detect_streamer_peaks(
    profile: AngularProfile,
    *,
    prominence_mad: float = 0.5,
    minimum_distance_deg: float = 15.0,
    smoothing_sigma_deg: float = 3.0,
) -> np.ndarray:
    """Detect peaks on a tripled profile so 0/360 is treated continuously."""
    if profile.status != "ok":
        return np.array([], dtype=float)
    step = float(profile.pa_deg[1] - profile.pa_deg[0])
    smooth = smooth_circular_profile(
        profile.values, sigma_deg=smoothing_sigma_deg, pa_step_deg=step
    )
    if not np.isfinite(smooth).all():
        indices = np.arange(smooth.size)
        finite = np.isfinite(smooth)
        if finite.sum() < 3:
            return np.array([], dtype=float)
        extended_x = np.concatenate((indices[finite] - smooth.size, indices[finite], indices[finite] + smooth.size))
        extended_y = np.tile(smooth[finite], 3)
        smooth = np.interp(indices, extended_x, extended_y)
    tripled = np.tile(smooth, 3)
    peaks, _ = find_peaks(
        tripled,
        prominence=prominence_mad,
        distance=max(1, round(minimum_distance_deg / step)),
    )
    middle = peaks[(peaks >= smooth.size) & (peaks < 2 * smooth.size)] - smooth.size
    return profile.pa_deg[middle]
