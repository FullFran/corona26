"""Phase C — magnetic topology across the ensemble.

    uv run python scripts/phase_c.py --file data/raw/<adapt>.fts.gz

Traces field lines from an equal-area photospheric grid, classifies them
open/closed, verifies the classification is converged, and produces figure 3.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from corona26.data.adapt import load_adapt, realisation_map, to_cea
from corona26.geometry import observer_geometry
from corona26.magnetic.pfss import RSS_ENSEMBLE, solve
from corona26.magnetic.trace import classification_is_converged, trace_topology
from corona26.plotting.magnetogram import limb_longitudes
from corona26.plotting.topology import plot_topology

warnings.filterwarnings("ignore", message=".*ran out of steps.*")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--nr", type=int, default=100)
    parser.add_argument("--n-lat", type=int, default=90)
    parser.add_argument("--n-lon", type=int, default=180)
    parser.add_argument("--realisations", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    adapt = load_adapt(args.file)
    n_real = args.realisations or adapt.n_realisations
    print(f"boundary : {adapt.path.name}  ({adapt.map_time.isot}Z)")
    print(f"seeds    : {args.n_lat} x {args.n_lon} = {args.n_lat * args.n_lon} "
          "equal-area photospheric seeds")
    print()

    cea_maps = [to_cea(realisation_map(adapt, i)) for i in range(n_real)]

    reference_output = solve(cea_maps[0], rss=2.5, nr=args.nr).output
    converged, changed = classification_is_converged(reference_output)
    print(f"classification converged : {converged}  "
          f"({changed} seeds change class when the step budget doubles)")
    if not converged:
        print("  WARNING: raise max_steps before trusting the open fraction")
    print()

    fractions: dict[float, list[float]] = {r: [] for r in RSS_ENSEMBLE}
    reference: dict[float, object] = {}
    t_start = time.time()

    for rss in RSS_ENSEMBLE:
        t0 = time.time()
        for i, cea in enumerate(cea_maps):
            out = solve(cea, rss=rss, nr=args.nr).output
            topo = trace_topology(out, n_lat=args.n_lat, n_lon=args.n_lon)
            fractions[rss].append(topo.open_area_fraction)
            if i == 0:
                reference[rss] = topo
        f = 100 * np.array(fractions[rss])
        print(f"  Rss={rss:4.1f}  open area {f.mean():5.2f}% "
              f"[{f.min():5.2f}, {f.max():5.2f}]   ({time.time() - t0:5.1f}s)")

    means = np.array([100 * np.mean(fractions[r]) for r in RSS_ENSEMBLE])
    within = np.mean([
        100 * (np.max(fractions[r]) - np.min(fractions[r]))
        for r in RSS_ENSEMBLE
    ])
    across = means.max() - means.min()
    print()
    print(f"open area spread within Rss (boundary) : {within:5.2f} percentage points")
    print(f"open area spread across Rss (model)    : {across:5.2f} percentage points")
    print(f"ratio                                  : {across / within:5.1f}x")

    topo = reference[2.5]
    print()
    print("at the conventional Rss = 2.5:")
    for k, v in topo.summary().items():
        print(f"  {k:28s} {v}")

    geom = observer_geometry()
    args.outdir.mkdir(parents=True, exist_ok=True)
    figure = plot_topology(
        topo, fractions, args.outdir / "topology.png", rss=2.5,
        map_time=f"ADAPT-GONG {adapt.map_time.isot}Z · realisation 0",
        limbs=limb_longitudes(geom.l0.value),
    )

    manifest = args.outdir / "topology.manifest.json"
    manifest.write_text(json.dumps({
        "map_file": adapt.path.name,
        "map_time_utc": adapt.map_time.isot,
        "seeds": args.n_lat * args.n_lon,
        "classification_converged": bool(converged),
        "seeds_changing_class_on_doubling": int(changed),
        "open_area_fraction": {str(r): float(np.mean(fractions[r]))
                               for r in RSS_ENSEMBLE},
        "open_area_spread_within_rss_points": round(float(within), 3),
        "open_area_spread_across_rss_points": round(float(across), 3),
        "at_rss_2.5": topo.summary(),
        "wall_seconds": round(time.time() - t_start, 1),
    }, indent=2) + "\n")

    print()
    print(f"figure   : {figure}")
    print(f"manifest : {manifest}")


if __name__ == "__main__":
    main()
