"""ADAPT-GONG global magnetic maps: the boundary condition for PFSS.

ADAPT (Air Force Data Assimilative Photospheric flux Transport) takes GONG
magnetograms and evolves them with a surface flux transport model to estimate
the hemisphere we cannot see. Each file ships 12 realisations produced by
varying the transport model's assumptions — a free, physically motivated
uncertainty ensemble on the single largest error source in this project.

Two properties of the format matter and are easy to get silently wrong:

1. The native grid is **plate carrée** (equal steps in latitude), but the PFSS
   solver requires **cylindrical equal area** (equal steps in sin(latitude)).
   Feeding a CAR map to the solver as if it were CEA over-weights the poles
   and produces a plausible, wrong corona. `to_cea` handles the reprojection.

2. Surface integrals over a CAR grid must be weighted by cos(latitude),
   because equal-angle cells do not have equal area. `flux_balance` does this;
   forgetting it makes the monopole diagnostic meaningless.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import astropy.units as u
import numpy as np
import sunpy.map
from astropy.io import fits
from astropy.time import Time

# ADAPT fills unobserved/invalid pixels with this sentinel.
MISSING_VALUE = -9999.0

# Latitude of the first and last row centres on the 180-row native grid.
_DEG = u.deg


@dataclass(frozen=True)
class AdaptMap:
    """One ADAPT file: 12 realisations plus their spread and provenance."""

    data: np.ndarray  # (n_realisations, n_lat, n_lon), Gauss
    stddev: np.ndarray | None  # (n_lat, n_lon), Gauss, spread across realisations
    header: fits.Header
    path: Path

    @property
    def n_realisations(self) -> int:
        return self.data.shape[0]

    @property
    def map_time(self) -> Time:
        """Observation time of the map — the timestamp that makes this a prediction."""
        return Time(self.header["MAPTIME"], scale="utc")

    @property
    def carrington_longitude(self) -> u.Quantity:
        """Carrington longitude of central meridian at map time."""
        return self.header["MAPLON"] * _DEG

    @property
    def assimilated_farside(self) -> bool:
        """Whether far-side data assimilation was applied.

        ADAPT records ``-9999`` when no far-side assimilation has happened.
        When this is False the far side is a flux-transport extrapolation,
        which is the dominant uncertainty in the whole pipeline.
        """
        return self.header.get("LAST_FAR", MISSING_VALUE) != MISSING_VALUE


def _clean(data: np.ndarray) -> np.ndarray:
    """Replace ADAPT's missing sentinel with NaN."""
    out = np.asarray(data, dtype=np.float64).copy()
    out[out == MISSING_VALUE] = np.nan
    return out


def load_adapt(path: str | Path) -> AdaptMap:
    """Read an ADAPT FITS file into memory, sentinel values cleaned to NaN."""
    path = Path(path)
    with fits.open(path) as hdul:
        header = hdul[0].header.copy()
        data = _clean(hdul[0].data)
        stddev = None
        for hdu in hdul[1:]:
            if hdu.name.upper() == "STDDEV":
                stddev = _clean(hdu.data)
                break
    if data.ndim != 3:
        raise ValueError(f"expected a 3-D ADAPT cube, got shape {data.shape}")
    return AdaptMap(data=data, stddev=stddev, header=header, path=path)


def latitude_centres(n_lat: int) -> np.ndarray:
    """Latitudes (degrees) of row centres on ADAPT's equal-angle grid."""
    edges = np.linspace(-90.0, 90.0, n_lat + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def flux_balance(br: np.ndarray) -> dict[str, float]:
    """Area-weighted magnetic flux diagnostics for one realisation.

    A global radial field must satisfy ``∮ Br dA = 0``. It never does exactly —
    ADAPT damps but does not eliminate the residual monopole — but a large
    imbalance means the map is wrong, and PFSS will happily solve it anyway
    and hand back a corona with a spurious global structure.

    Returns signed flux, unsigned flux, and their ratio. The ratio is the
    number to watch: a few percent is normal, tens of percent is a red flag.
    """
    br = np.asarray(br, dtype=np.float64)
    if br.ndim != 2:
        raise ValueError(f"expected a 2-D (lat, lon) map, got shape {br.shape}")

    lat = np.deg2rad(latitude_centres(br.shape[0]))
    weights = np.cos(lat)[:, None] * np.ones((1, br.shape[1]))

    valid = np.isfinite(br)
    w = np.where(valid, weights, 0.0)
    values = np.where(valid, br, 0.0)

    signed = float(np.sum(values * w))
    unsigned = float(np.sum(np.abs(values) * w))
    ratio = abs(signed) / unsigned if unsigned > 0 else 0.0
    return {
        "signed_flux": signed,
        "unsigned_flux": unsigned,
        "monopole_ratio": ratio,
    }


def realisation_map(adapt: AdaptMap, index: int) -> sunpy.map.GenericMap:
    """Build a sunpy map in Carrington CAR projection for one realisation.

    ADAPT's native header declares ``WCSNAME = 'Heliocentric-cart'`` with
    ``CTYPE = 'Long'/'Lat'``, which is not valid WCS. We rewrite it into a
    proper Carrington plate-carrée header so downstream coordinate handling
    is not guessing.
    """
    if not 0 <= index < adapt.n_realisations:
        raise IndexError(
            f"realisation {index} out of range (0..{adapt.n_realisations - 1})"
        )

    data = adapt.data[index]
    n_lat, n_lon = data.shape

    header = sunpy.map.make_fitswcs_header(
        data,
        sunpy.coordinates.frames.HeliographicCarrington(
            0 * _DEG,
            0 * _DEG,
            obstime=adapt.map_time,
            observer="earth",
        ),
        scale=[360 / n_lon, 180 / n_lat] * _DEG / u.pix,
        projection_code="CAR",
    )
    header["bunit"] = "G"
    return sunpy.map.Map(data, header)


def to_cea(car_map: sunpy.map.GenericMap) -> sunpy.map.GenericMap:
    """Reproject a plate-carrée map to cylindrical equal area for the solver.

    The PFSS solver needs values equally spaced in sin(latitude). Skipping
    this step does not raise — it silently over-weights the poles.
    """
    from sunkit_magex.pfss import utils

    return utils.car_to_cea(car_map)


def write_manifest(path: str | Path, adapt: AdaptMap, **extra: object) -> Path:
    """Record what this boundary condition actually was.

    A prediction whose input magnetogram time is unknown is worthless, so
    every artefact derived from this map carries the provenance alongside it.
    """
    path = Path(path)
    manifest = {
        "source": "ADAPT-GONG",
        "file": adapt.path.name,
        "map_time_utc": adapt.map_time.isot,
        "carrington_longitude_deg": float(adapt.carrington_longitude.to_value(_DEG)),
        "carrington_rotation": float(adapt.header.get("MAPCR", float("nan"))),
        "n_realisations": adapt.n_realisations,
        "units": adapt.header.get("BUNIT", "unknown"),
        "grid_shape": list(adapt.data.shape),
        "model_version": adapt.header.get("MODELVER", "unknown"),
        "input_magnetograph": adapt.header.get("MAPDATA", "unknown"),
        "mag_type": "line-of-sight" if adapt.header.get("MAG_TYPE") == 1 else "radial",
        "flux_scaling_applied": adapt.header.get("FLUXSCAL"),
        "monopole_damping_public": adapt.header.get("MONODPUB"),
        "residual_monopole": adapt.header.get("MONOFINL"),
        "farside_assimilated": adapt.assimilated_farside,
        "column_coverage_percent": adapt.header.get("COVERAGE"),
        "written_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        **extra,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def fetch_adapt(
    before: Time | str | None = None,
    *,
    window: u.Quantity = 12 * u.hour,
    outdir: str | Path = "data/raw",
) -> Path:
    """Download the most recent Carrington-fixed ADAPT-GONG map before a time.

    Carrington-fixed (``ADAPTLonType('0')``) is the right product for PFSS:
    the solver wants a global map in a fixed heliographic frame, not one
    re-centred on the current central meridian.
    """
    from sunpy.net import Fido
    from sunpy.net import attrs as a

    before = Time(before) if before is not None else Time.now()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = Fido.search(
        a.Time(before - window, before),
        a.Instrument("adapt"),
        a.adapt.ADAPTLonType("0"),
    )
    if len(results[0]) == 0:
        raise RuntimeError(
            f"no ADAPT maps found in the {window} before {before.isot}; "
            "widen the window or fall back to a GONG synoptic map"
        )

    files = Fido.fetch(results[0, -1], path=str(outdir / "{file}"))
    if not files:
        raise RuntimeError("ADAPT search succeeded but the download returned no files")
    return Path(files[0])
