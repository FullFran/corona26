"""Electron density modulated by magnetic topology.

The radial profile alone renders a boring circular halo. What makes an eclipse
corona look like an eclipse corona is that closed field traps plasma and open
field lets it escape:

* **closed** regions are overdense and bright — streamers, helmet streamers;
* **open** regions are underdense and dark — coronal holes.

So we trace a field line through every point of a 3-D grid and ask whether it
escapes. That is direct and uses only what PFSS already gives us, but it is
still a **proxy**: the enhancement factors below are parameters, not physics,
and they are fixed *before* any comparison with Predictive Science so that we
cannot quietly tune our way into agreement.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.ndimage import gaussian_filter

from corona26.magnetic.trace import DEFAULT_BATCH, _trace_in_batches
from corona26.plasma.radial_density import baumbach_allen

# Density contrast between trapped and escaping plasma. Observationally
# streamers run a few times denser than coronal holes at the same height
# (Saito, Poland & Munro 1977; Guhathakurta et al. 1996). These are the knobs.
CLOSED_ENHANCEMENT = 3.5
OPEN_DEPLETION = 0.4

# Smoothing applied to the closed/open indicator, in grid cells. A hard 0/1
# switch renders as unphysical faceting along the separatrix.
SMOOTHING_CELLS = (0.6, 1.0, 1.0)


@dataclass(frozen=True)
class DensityCube:
    """``ne`` sampled on a spherical grid, ready for line-of-sight lookup."""

    ne: np.ndarray            # (n_r, n_lat, n_lon) electrons cm^-3
    closedness: np.ndarray    # (n_r, n_lat, n_lon) in [0, 1]
    r: np.ndarray             # (n_r,) solar radii
    sin_lat: np.ndarray       # (n_lat,)
    lon: np.ndarray           # (n_lon,) degrees
    rss: float

    @property
    def r_max(self) -> float:
        return float(self.r[-1])

    def summary(self) -> dict[str, float]:
        return {
            "grid": list(self.ne.shape),
            "r_range": [float(self.r[0]), float(self.r[-1])],
            "closed_volume_fraction": float(np.mean(self.closedness > 0.5)),
            "closed_enhancement": CLOSED_ENHANCEMENT,
            "open_depletion": OPEN_DEPLETION,
            "rss": self.rss,
        }


def spherical_grid(
    *, n_r: int = 32, n_lat: int = 48, n_lon: int = 96,
    r_min: float = 1.02, r_max: float = 3.0,
):
    """Radii, sin(latitude) and longitude axes for the density cube.

    Radii are spaced logarithmically because the density falls by four orders
    of magnitude across this range; linear spacing wastes samples far out where
    nothing changes and starves the region near the limb where everything does.
    """
    r = np.geomspace(r_min, r_max, n_r)
    sin_lat_edges = np.linspace(-1.0, 1.0, n_lat + 1)
    sin_lat = 0.5 * (sin_lat_edges[:-1] + sin_lat_edges[1:])
    lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)
    return r, sin_lat, lon


def closedness_cube(
    output,
    r: np.ndarray,
    sin_lat: np.ndarray,
    lon: np.ndarray,
    rss: float,
    *,
    max_steps: int = 3000,
    batch: int = DEFAULT_BATCH,
    smoothing=SMOOTHING_CELLS,
    progress: bool = False,
) -> np.ndarray:
    """Fraction-of-closedness at every grid point, in [0, 1].

    A field line is traced through each grid point; 1 means it returns to the
    surface at both ends (plasma trapped), 0 means it reaches the source
    surface (plasma escapes). The result is smoothed so the separatrix is a
    gradient rather than a staircase.

    This traces one line per grid cell, so a 40x64x128 cube is 327,680 lines.
    The compiled tracer preallocates ``n_seeds * max_steps`` buffers, so doing
    that in one call asks for tens of gigabytes; `_trace_in_batches` keeps peak
    memory flat instead. See `corona26.magnetic.trace`.
    """
    # The PFSS domain stops at the source surface, so only shells inside it can
    # be traced. Above it the field is radial by construction, which means a
    # field line at a given latitude and longitude stays at that latitude and
    # longitude: the structure propagates straight out. Copying the outermost
    # traced shell outward is therefore not a fudge, it is what the model says.
    # Truncating instead would put a hard edge across the middle of the image.
    traced_mask = r <= rss * 0.999
    if not np.any(traced_mask):
        raise ValueError("no grid shells lie inside the source surface")
    r_traced = r[traced_mask]

    r_grid, lat_grid, lon_grid = np.meshgrid(
        r_traced, np.degrees(np.arcsin(sin_lat)), lon, indexing="ij"
    )
    seeds = SkyCoord(
        lon_grid.ravel() * u.deg,
        lat_grid.ravel() * u.deg,
        r_grid.ravel() * u.R_sun,
        frame=output.coordinate_frame,
    )

    flags = []
    total = len(seeds)
    for i, (is_open, _, _) in enumerate(
        _trace_in_batches(output, seeds, max_steps=max_steps, batch=batch)
    ):
        flags.append(np.where(is_open, 0.0, 1.0))
        if progress:
            done = min((i + 1) * batch, total)
            print(f"\r  tracing {done}/{total} lines", end="", flush=True)
    if progress:
        print()
    traced = np.concatenate(flags).reshape(r_grid.shape)

    closed = np.empty((r.size, sin_lat.size, lon.size))
    closed[traced_mask] = traced
    closed[~traced_mask] = traced[-1]  # radial propagation above the source surface

    if smoothing:
        # Longitude wraps; pad it so the seam does not get smoothed against
        # empty space and leave a visible ridge at zero degrees.
        padded = np.concatenate([closed[:, :, -8:], closed, closed[:, :, :8]], axis=2)
        padded = gaussian_filter(padded, sigma=smoothing, mode="nearest")
        closed = padded[:, :, 8:-8]

    return np.clip(closed, 0.0, 1.0)


def build_density_cube(
    output,
    *,
    n_r: int = 32,
    n_lat: int = 48,
    n_lon: int = 96,
    r_max: float = 6.0,
    rss: float = 2.5,
    closed_enhancement: float = CLOSED_ENHANCEMENT,
    open_depletion: float = OPEN_DEPLETION,
    batch: int = DEFAULT_BATCH,
    progress: bool = False,
) -> DensityCube:
    """Assemble ``ne(r, lat, lon)`` from the radial profile and the topology."""
    r, sin_lat, lon = spherical_grid(
        n_r=n_r, n_lat=n_lat, n_lon=n_lon, r_max=r_max
    )
    closed = closedness_cube(
        output, r, sin_lat, lon, rss, batch=batch, progress=progress
    )

    radial = baumbach_allen(r)[:, None, None]
    modulation = open_depletion + (closed_enhancement - open_depletion) * closed
    return DensityCube(
        ne=radial * modulation,
        closedness=closed,
        r=r,
        sin_lat=sin_lat,
        lon=lon,
        rss=rss,
    )


def sample_density(cube: DensityCube, r, sin_lat, lon_deg) -> np.ndarray:
    """Trilinear lookup into the cube, with longitude wrapping.

    Points outside the radial range return zero: below ``r[0]`` we are inside
    the occulted disk, and above ``r[-1]`` the model has nothing to say.
    """
    r = np.asarray(r, dtype=np.float64)
    sin_lat = np.clip(np.asarray(sin_lat, dtype=np.float64), -1.0, 1.0)
    lon_deg = np.mod(np.asarray(lon_deg, dtype=np.float64), 360.0)

    inside = (r >= cube.r[0]) & (r <= cube.r[-1])

    # Fractional indices along each axis.
    fr = np.interp(r, cube.r, np.arange(cube.r.size))
    fs = np.interp(sin_lat, cube.sin_lat, np.arange(cube.sin_lat.size))
    dlon = 360.0 / cube.lon.size
    fl = lon_deg / dlon

    i0 = np.clip(np.floor(fr).astype(int), 0, cube.r.size - 2)
    j0 = np.clip(np.floor(fs).astype(int), 0, cube.sin_lat.size - 2)
    k0 = np.floor(fl).astype(int) % cube.lon.size
    k1 = (k0 + 1) % cube.lon.size

    tr = np.clip(fr - i0, 0.0, 1.0)[..., None, None]
    ts = np.clip(fs - j0, 0.0, 1.0)[..., None]
    tl = np.clip(fl - k0, 0.0, 1.0)

    ne = cube.ne
    c00 = ne[i0, j0, k0] * (1 - tl) + ne[i0, j0, k1] * tl
    c01 = ne[i0, j0 + 1, k0] * (1 - tl) + ne[i0, j0 + 1, k1] * tl
    c10 = ne[i0 + 1, j0, k0] * (1 - tl) + ne[i0 + 1, j0, k1] * tl
    c11 = ne[i0 + 1, j0 + 1, k0] * (1 - tl) + ne[i0 + 1, j0 + 1, k1] * tl

    ts = ts[..., 0]
    tr = tr[..., 0, 0]
    c0 = c00 * (1 - ts) + c01 * ts
    c1 = c10 * (1 - ts) + c11 * ts
    return np.where(inside, c0 * (1 - tr) + c1 * tr, 0.0)
