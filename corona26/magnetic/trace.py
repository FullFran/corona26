"""Magnetic topology: which field lines escape, and which trap plasma.

Field lines are integrated from photospheric seeds and classified:

* **closed** — both footpoints return to the surface. Plasma is trapped and
  overdense. These build streamers and helmet streamers.
* **open** — the line reaches the source surface. Plasma escapes as solar
  wind and the region is underdense. These are coronal holes, dark in white
  light.

That classification is what turns a magnetic field into something we can put a
density on, so it feeds directly into the electron-density proxy in
``corona26.plasma``.

Two things here are easy to get silently wrong:

1. **Seeding must be equal-area.** Uniform steps in latitude over-sample the
   poles, so an "open fraction" computed from such a grid is not an area
   fraction and cannot be compared to published coronal-hole coverage.
   `photospheric_seeds` samples uniformly in sin(latitude).

2. **Integration must be converged.** A line that runs out of steps before
   returning to the surface is indistinguishable from one that escaped.
   `classification_is_converged` checks that doubling the step budget changes
   no classifications.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from sunkit_magex.pfss import tracing

# Seeds sit just above the boundary; starting exactly on it makes the
# interpolation ambiguous at the first step.
SEED_RADIUS = 1.01 * u.R_sun

DEFAULT_MAX_STEPS = 5000

# The compiled tracer preallocates one buffer of (n_seeds, max_steps, 3)
# doubles *per call*, several times over. Memory is therefore the product of
# the seed count and the step budget, not a function of either alone: 300k
# seeds at 3000 steps asks for tens of gigabytes before a single line is
# traced. Tracing in batches makes peak memory constant and costs nothing,
# because every batch is independent anyway.
#
# 4000 seeds x 5000 steps x 3 x 8 bytes x ~3 buffers is roughly 1.4 GB, which
# is the most we want to see on a laptop.
DEFAULT_BATCH = 4000


def _trace_in_batches(output, seeds, *, max_steps: int, batch: int = DEFAULT_BATCH):
    """Trace seeds in fixed-size batches, keeping only what we need.

    Yields ``(is_open, polarity, expansion_factor)`` arrays per batch. The
    field-line geometry is discarded as soon as each batch is classified,
    which is what keeps peak memory flat as the seed count grows.
    """
    for start in range(0, len(seeds), batch):
        chunk = seeds[start : start + batch]
        lines = tracing.PerformanceTracer(max_steps=max_steps).trace(chunk, output)
        yield (
            np.array([bool(f.is_open) for f in lines]),
            np.array([int(f.polarity) for f in lines]),
            np.array(
                [float(f.expansion_factor) if f.is_open else np.nan for f in lines]
            ),
        )
        del lines


@dataclass(frozen=True)
class Topology:
    """Open/closed classification on an equal-area photospheric grid."""

    is_open: np.ndarray        # (n_lat, n_lon) bool
    polarity: np.ndarray       # (n_lat, n_lon) -1, 0 or +1
    expansion_factor: np.ndarray  # (n_lat, n_lon) float, NaN where closed
    n_lat: int
    n_lon: int

    @property
    def open_area_fraction(self) -> float:
        """Fraction of the solar surface covered by open field.

        Because the grid is uniform in sin(latitude), every cell subtends the
        same solid angle and this is a plain mean — no cos(lat) weighting.
        """
        return float(np.mean(self.is_open))

    def summary(self) -> dict[str, float]:
        open_ef = self.expansion_factor[self.is_open]
        return {
            "open_area_fraction": self.open_area_fraction,
            "n_seeds": int(self.is_open.size),
            "positive_open_fraction": float(
                np.mean(self.polarity[self.is_open] > 0)
            ) if self.is_open.any() else 0.0,
            "median_expansion_factor": float(np.nanmedian(open_ef))
            if open_ef.size else float("nan"),
        }


def photospheric_seeds(
    frame,
    *,
    n_lat: int = 60,
    n_lon: int = 120,
    radius: u.Quantity = SEED_RADIUS,
) -> SkyCoord:
    """An equal-area grid of seed points just above the photosphere.

    Uniform in sin(latitude), so every seed represents the same surface area
    and open-field statistics are area fractions by construction.
    """
    sin_lat_edges = np.linspace(-1.0, 1.0, n_lat + 1)
    sin_lat = 0.5 * (sin_lat_edges[:-1] + sin_lat_edges[1:])
    lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)

    lon_grid, sin_lat_grid = np.meshgrid(lon, sin_lat)
    lat_grid = np.degrees(np.arcsin(sin_lat_grid))

    return SkyCoord(
        lon_grid.ravel() * u.deg,
        lat_grid.ravel() * u.deg,
        radius,
        frame=frame,
    )


def trace_topology(
    output,
    *,
    n_lat: int = 60,
    n_lon: int = 120,
    max_steps: int = DEFAULT_MAX_STEPS,
    batch: int = DEFAULT_BATCH,
) -> Topology:
    """Trace from an equal-area photospheric grid and classify every line."""
    seeds = photospheric_seeds(output.coordinate_frame, n_lat=n_lat, n_lon=n_lon)

    parts = list(_trace_in_batches(output, seeds, max_steps=max_steps, batch=batch))
    shape = (n_lat, n_lon)
    is_open = np.concatenate([p[0] for p in parts]).reshape(shape)
    polarity = np.concatenate([p[1] for p in parts]).reshape(shape)
    expansion = np.concatenate([p[2] for p in parts]).reshape(shape)

    return Topology(
        is_open=is_open,
        polarity=polarity,
        expansion_factor=expansion,
        n_lat=n_lat,
        n_lon=n_lon,
    )


def classification_is_converged(
    output,
    *,
    n_lat: int = 40,
    n_lon: int = 80,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> tuple[bool, int]:
    """Does doubling the integration budget change any classification?

    A field line that exhausts its step budget mid-flight looks exactly like
    one that never closed. Returns ``(converged, n_changed)``.
    """
    coarse = trace_topology(output, n_lat=n_lat, n_lon=n_lon, max_steps=max_steps)
    fine = trace_topology(output, n_lat=n_lat, n_lon=n_lon, max_steps=2 * max_steps)
    changed = int(np.sum(coarse.is_open != fine.is_open))
    return changed == 0, changed


def sample_field_lines(
    output,
    *,
    n_lat: int = 18,
    n_lon: int = 36,
    max_steps: int = DEFAULT_MAX_STEPS,
):
    """A sparse set of traced lines for 3-D visualisation."""
    seeds = photospheric_seeds(output.coordinate_frame, n_lat=n_lat, n_lon=n_lon)
    return tracing.PerformanceTracer(max_steps=max_steps).trace(seeds, output)
