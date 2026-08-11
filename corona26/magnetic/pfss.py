"""Potential field source surface reconstruction.

Assumes a current-free corona between the photosphere and a source surface at
``Rss``, where the field is forced radial to mimic the solar wind dragging it
open. See ``docs/physics.md`` §4 for the derivation and for what this buys and
costs.

The dangerous failure here is silent. A sign convention error, or feeding the
solver a plate-carrée map when it expects equal-area, does not raise: it
returns a plausible corona that is wrong. Hence `dipole_br`, which has a closed
form the solver must reproduce, and `boundary_residual`, which checks the
solution against the map it came from.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
import sunpy.map
from sunkit_magex import pfss as _pfss

# The historical default (Altschuler & Newkirk 1969). A convention, not a
# measurement — which is why we run an ensemble rather than trusting it.
DEFAULT_RSS = 2.5

# Source-surface radii spanning the range recent eclipse-benchmarked work finds
# plausible across the solar cycle.
RSS_ENSEMBLE = (1.3, 1.5, 2.0, 2.5, 3.0)


@dataclass(frozen=True)
class Solution:
    """A solved PFSS field plus the parameters that produced it."""

    output: _pfss.Output
    rss: float
    nr: int
    realisation: int

    @property
    def source_surface_br(self) -> sunpy.map.GenericMap:
        return self.output.source_surface_br


def dipole_br(
    theta: np.ndarray, r: np.ndarray, rss: float, b0: float = 1.0
) -> np.ndarray:
    """Analytic PFSS radial field for a pure dipole boundary condition.

    For ``Br(R☉, θ) = b0 cos θ`` only the ``l = 1`` harmonic survives, and the
    potential reduces to ``Φ = (a r + b / r²) cos θ``. Applying
    ``Bθ(Rss) = 0`` gives ``a = -b / Rss³``, and ``Br(1) = b0`` then fixes
    ``b = b0 / (2 + Rss⁻³)``, so

        Br(r, θ) = b0 cos θ · (Rss⁻³ + 2 r⁻³) / (2 + Rss⁻³)

    This is the reference the numerical solver must reproduce. It pins the sign
    convention, the normalisation and the upper boundary condition at once.

    Parameters
    ----------
    theta
        Colatitude in radians.
    r
        Heliocentric radius in solar radii.
    rss
        Source surface radius in solar radii.
    """
    r = np.asarray(r, dtype=np.float64)
    if np.any(r < 1.0) or np.any(r > rss):
        raise ValueError("r must lie between 1 and rss")
    b = b0 / (2.0 + rss**-3)
    return np.cos(theta) * b * (rss**-3 + 2.0 * r**-3)


def dipole_map(
    n_lat: int = 180, n_lon: int = 360, b0: float = 1.0, obstime: str = "2026-08-11"
) -> sunpy.map.GenericMap:
    """A pure dipole boundary condition on the equal-area grid the solver wants.

    The CEA grid is uniform in sin(latitude), so the dipole ``cos θ = sin(lat)``
    is uniform in the row index — a useful property to build directly rather
    than reproject.
    """
    from sunkit_magex.pfss import utils

    sin_lat_edges = np.linspace(-1, 1, n_lat + 1)
    sin_lat = 0.5 * (sin_lat_edges[:-1] + sin_lat_edges[1:])
    data = b0 * np.repeat(sin_lat[:, None], n_lon, axis=1)
    header = utils.carr_cea_wcs_header(obstime, (n_lon, n_lat))
    return sunpy.map.Map(data, header)


def solve(
    br_cea: sunpy.map.GenericMap,
    *,
    rss: float = DEFAULT_RSS,
    nr: int = 100,
    realisation: int = 0,
) -> Solution:
    """Solve PFSS for one equal-area boundary map.

    ``br_cea`` must already be in a CEA (equal sin-latitude) projection. Pass a
    plate-carrée map and the solver will not complain — it will over-weight the
    poles and return a wrong answer. Use `corona26.data.adapt.to_cea` first.
    """
    ctype = str(br_cea.meta.get("ctype2", "")).upper()
    if "CEA" not in ctype:
        raise ValueError(
            f"expected a CEA projection for the PFSS solver, got ctype2={ctype!r}; "
            "reproject with corona26.data.adapt.to_cea first"
        )
    output = _pfss.pfss(_pfss.Input(br_cea, nr, rss))
    return Solution(output=output, rss=rss, nr=nr, realisation=realisation)


def solved_boundary_br(solution: Solution) -> np.ndarray:
    """Br at the photospheric boundary, on the same cell centres as the input.

    The solver stores two staggered representations. ``bg`` lives on cell
    *corners* — shape ``(nlon+1, nlat+1, nr+1, 3)`` — while ``bc[0]`` is the
    radial component on cell *centres*, ``(nlon, nlat, nr+1)``, which is where
    the input map's pixels are. Comparing ``bg`` against the input is an
    off-by-half-a-cell error that shows up as a ~27% residual and looks
    convincingly like a truncation artefact. Use ``bc``.
    """
    return np.asarray(solution.output.bc[0], dtype=np.float64)[:, :, 0].T


def boundary_residual(solution: Solution, br_cea: sunpy.map.GenericMap) -> float:
    """Relative RMS mismatch between the solved inner boundary and the input.

    Should be very small — the solver ingests the boundary essentially exactly.
    A large residual means the map was not read the way we think it was:
    wrong projection, wrong transpose, or wrong staggering.
    """
    solved = solved_boundary_br(solution)
    target = np.asarray(br_cea.data, dtype=np.float64)
    if solved.shape != target.shape:
        raise ValueError(
            f"boundary shape {solved.shape} does not match input {target.shape}"
        )
    rms = float(np.sqrt(np.nanmean((solved - target) ** 2)))
    scale = float(np.sqrt(np.nanmean(target**2)))
    return rms / scale if scale > 0 else np.inf


def open_flux(solution: Solution) -> float:
    """Unsigned magnetic flux threading the source surface, in G R☉².

    All field lines reaching the source surface are open, so this is the total
    flux carried into the heliosphere. It is the single number most sensitive
    to the choice of ``Rss``, and therefore the cleanest scalar summary of what
    the ensemble disagrees about.
    """
    ss = solution.source_surface_br
    data = np.asarray(ss.data, dtype=np.float64)
    # The source-surface map is CEA: rows are equal steps in sin(latitude), so
    # every cell subtends the same solid angle and no cos(lat) weight is needed.
    n_lat, n_lon = data.shape
    cell = (4.0 * np.pi / (n_lat * n_lon)) * solution.rss**2
    return float(np.nansum(np.abs(data)) * cell)


def cache_key(
    map_file: str, realisation: int, rss: float, nr: int, version: str = "1"
) -> str:
    """Content hash for a solve, so the ensemble is never recomputed twice."""
    payload = f"{map_file}|{realisation}|{rss:.4f}|{nr}|{version}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_path(cache_dir: str | Path, key: str) -> Path:
    return Path(cache_dir) / f"pfss_{key}.npz"
