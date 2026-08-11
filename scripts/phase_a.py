"""Phase A — fetch the boundary condition and check it is usable.

    uv run python scripts/phase_a.py

Downloads the most recent Carrington-fixed ADAPT-GONG map, reports the flux
diagnostics that reveal a bad map before PFSS silently solves it, writes the
provenance manifest, and produces figure 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from corona26.data.adapt import fetch_adapt, load_adapt, write_manifest
from corona26.geometry import observer_geometry
from corona26.plotting.magnetogram import (
    ensemble_summary,
    limb_longitudes,
    plot_magnetogram,
    spread_vs_longitude,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, help="use a local ADAPT file instead of downloading")
    parser.add_argument("--realisation", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    path = args.file or fetch_adapt()
    print(f"boundary condition : {path.name}")

    adapt = load_adapt(path)
    print(f"map time           : {adapt.map_time.isot}Z")
    print(f"realisations       : {adapt.n_realisations}")
    print(f"grid               : {adapt.data.shape[1]} lat x {adapt.data.shape[2]} lon")
    print(f"CM Carrington lon  : {adapt.carrington_longitude.value:.2f} deg")
    print(f"far-side assimilated: {adapt.assimilated_farside}")

    summary = ensemble_summary(adapt)
    print()
    print("flux diagnostics across the ensemble")
    print(f"  monopole ratio      : {summary['monopole_ratio_min']:.4f} .. "
          f"{summary['monopole_ratio_max']:.4f}  (mean {summary['monopole_ratio_mean']:.4f})")
    print(f"  unsigned flux spread: {summary['unsigned_flux_spread_percent']:.2f} % "
          "across realisations")

    geom = observer_geometry()
    limb_spread: dict[str, float] = {}
    if adapt.stddev is not None:
        profile = spread_vs_longitude(adapt)
        print()
        print(f"boundary uncertainty at totality (L0 = {geom.l0.value:.2f} deg)")
        for name, lon in limb_longitudes(geom.l0.value).items():
            band = np.roll(profile, -int(lon) + 20)[:41].mean()  # +/-20 deg
            limb_spread[f"sigma_{name.replace(' ', '_')}_G"] = float(band)
            print(f"  {name:12s} lon {lon:6.1f} deg   mean sigma = {band:5.2f} G")

    args.outdir.mkdir(parents=True, exist_ok=True)
    figure = plot_magnetogram(
        adapt,
        args.outdir / "magnetogram.png",
        realisation=args.realisation,
        l0_at_totality=geom.l0.value,
    )
    manifest = write_manifest(
        args.outdir / "magnetogram.manifest.json",
        adapt,
        observer_geometry=geom.summary(),
        **summary,
        **limb_spread,
    )
    print()
    print(f"figure   : {figure}")
    print(f"manifest : {manifest}")


if __name__ == "__main__":
    main()
