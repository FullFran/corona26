"""Phase B — PFSS across the full ensemble.

    uv run python scripts/phase_b.py --file data/raw/<adapt>.fts.gz

Solves the potential field for every ADAPT realisation at every source-surface
radius, reports the boundary residual and open flux, and produces figure 2.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from corona26.data.adapt import load_adapt, realisation_map, to_cea
from corona26.magnetic.pfss import RSS_ENSEMBLE, boundary_residual, open_flux, solve
from corona26.plotting.pfss_ensemble import neutral_line_latitude, plot_rss_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--nr", type=int, default=100)
    parser.add_argument("--realisations", type=int, default=None,
                        help="limit for a fast smoke run")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    adapt = load_adapt(args.file)
    n_real = args.realisations or adapt.n_realisations
    print(f"boundary : {adapt.path.name}  ({adapt.map_time.isot}Z)")
    print(f"ensemble : {n_real} realisations x {len(RSS_ENSEMBLE)} source surfaces "
          f"= {n_real * len(RSS_ENSEMBLE)} solves, nr={args.nr}")
    print()

    cea_maps = [to_cea(realisation_map(adapt, i)) for i in range(n_real)]

    fluxes: dict[float, list[float]] = {r: [] for r in RSS_ENSEMBLE}
    residuals: list[float] = []
    reference: dict[float, np.ndarray] = {}
    neutral_lines: dict[float, list[np.ndarray]] = {r: [] for r in RSS_ENSEMBLE}

    t_start = time.time()
    for rss in RSS_ENSEMBLE:
        t0 = time.time()
        for i, cea in enumerate(cea_maps):
            sol = solve(cea, rss=rss, nr=args.nr, realisation=i)
            fluxes[rss].append(open_flux(sol))
            ss = np.asarray(sol.source_surface_br.data, dtype=np.float64)
            neutral_lines[rss].append(neutral_line_latitude(ss))
            if i == 0:
                reference[rss] = ss
                residuals.append(boundary_residual(sol, cea))
        f = np.array(fluxes[rss])
        print(f"  Rss={rss:4.1f}  open flux {f.mean():9.3f} "
              f"[{f.min():9.3f}, {f.max():9.3f}]  "
              f"spread {100 * (f.max() - f.min()) / f.mean():5.2f}%   "
              f"({time.time() - t0:.1f}s)")

    print()
    print(f"boundary residual (realisation 0): "
          f"{min(residuals):.4f} .. {max(residuals):.4f}")

    means = np.array([np.mean(fluxes[r]) for r in RSS_ENSEMBLE])
    within = np.mean([
        100 * (np.max(fluxes[r]) - np.min(fluxes[r])) / np.mean(fluxes[r])
        for r in RSS_ENSEMBLE
    ])
    across = 100 * (means.max() - means.min()) / means.mean()
    print(f"open flux spread within Rss (boundary uncertainty) : {within:6.2f}%")
    print(f"open flux spread across Rss (model uncertainty)    : {across:6.2f}%")
    print(f"ratio                                              : {across / within:6.1f}x")

    # How far the streamer belt moves when Rss changes, vs when the boundary does.
    belt_across = np.nanmean([
        np.nanmax([np.nanmean(neutral_lines[r][0]) for r in RSS_ENSEMBLE])
        - np.nanmin([np.nanmean(neutral_lines[r][0]) for r in RSS_ENSEMBLE])
    ])
    belt_rms = {
        r: float(np.nanstd(np.array(neutral_lines[r]), axis=0).mean())
        for r in RSS_ENSEMBLE
    }
    print()
    print("neutral-line latitude scatter across realisations [deg]:")
    for r in RSS_ENSEMBLE:
        print(f"  Rss={r:4.1f}  {belt_rms[r]:5.2f}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    figure = plot_rss_ensemble(
        reference, fluxes, args.outdir / "pfss_ensemble.png",
        map_time=f"ADAPT-GONG {adapt.map_time.isot}Z · realisation 0",
    )
    summary = {
        "map_file": adapt.path.name,
        "map_time_utc": adapt.map_time.isot,
        "n_realisations": n_real,
        "rss_values": list(RSS_ENSEMBLE),
        "nr": args.nr,
        "solves": n_real * len(RSS_ENSEMBLE),
        "wall_seconds": round(time.time() - t_start, 1),
        "boundary_residual_max": max(residuals),
        "open_flux_mean": {str(r): float(np.mean(fluxes[r])) for r in RSS_ENSEMBLE},
        "open_flux_spread_within_rss_percent": round(float(within), 3),
        "open_flux_spread_across_rss_percent": round(float(across), 3),
        "neutral_line_scatter_deg": {str(r): belt_rms[r] for r in RSS_ENSEMBLE},
    }
    manifest = args.outdir / "pfss_ensemble.manifest.json"
    manifest.write_text(json.dumps(summary, indent=2) + "\n")
    print()
    print(f"figure   : {figure}")
    print(f"manifest : {manifest}")
    print(f"total    : {summary['wall_seconds']}s")


if __name__ == "__main__":
    main()
