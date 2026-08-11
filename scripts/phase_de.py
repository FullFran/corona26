"""Phases D and E — electron density and the Thomson-scattering render.

    uv run python scripts/phase_de.py --file data/raw/<adapt>.fts.gz

Produces the prediction: what the corona should look like from Colmenar Viejo
at 20:31 CEST on 12 August 2026.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np

from corona26.data.adapt import load_adapt, realisation_map, to_cea
from corona26.geometry import observer_geometry
from corona26.magnetic.pfss import solve
from corona26.plasma.topology_density import build_density_cube
from corona26.plotting.corona import plot_channels, plot_corona
from corona26.radiation.camera import camera_for_totality, solar_north_from_zenith
from corona26.radiation.render import render

warnings.filterwarnings("ignore", message=".*ran out of steps.*")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--rss", type=float, default=2.5)
    parser.add_argument("--realisation", type=int, default=0)
    parser.add_argument("--pixels", type=int, default=900)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--fov", type=float, default=3.0)
    parser.add_argument("--nr-cube", type=int, default=48)
    parser.add_argument("--nlat", type=int, default=96)
    parser.add_argument("--nlon", type=int, default=192)
    parser.add_argument("--tile", type=int, default=64,
                        help="pixels per tile; peak memory scales as tile^2 * samples")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--memory-cap-gb", type=float, default=6.0,
        help="hard address-space limit; exceeding it raises MemoryError rather "
             "than letting the kernel OOM-kill the shell (0 disables)",
    )
    args = parser.parse_args()

    if args.memory_cap_gb > 0:
        cap = int(args.memory_cap_gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    adapt = load_adapt(args.file)
    geom = observer_geometry()
    print(f"boundary : {adapt.path.name}  ({adapt.map_time.isot}Z)")
    print(f"Rss      : {args.rss}   realisation {args.realisation}")
    print()

    t0 = time.time()
    output = solve(
        to_cea(realisation_map(adapt, args.realisation)), rss=args.rss, nr=100
    ).output
    print(f"PFSS solved                 {time.time() - t0:6.1f}s")

    t0 = time.time()
    cube = build_density_cube(
        output, n_r=args.nr_cube, n_lat=args.nlat, n_lon=args.nlon,
        rss=args.rss, progress=True
    )
    print(f"density cube built          {time.time() - t0:6.1f}s   "
          f"{cube.ne.shape}  closed volume "
          f"{100 * cube.summary()['closed_volume_fraction']:.1f}%")

    cam_north = camera_for_totality(args.pixels, args.fov)
    cam_horizon = camera_for_totality(args.pixels, args.fov, horizon_referenced=True)

    t0 = time.time()
    north = render(cube, cam_north, n_samples=args.samples, tile=args.tile)
    print(f"render north-up             {time.time() - t0:6.1f}s   "
          f"{args.pixels}px x {args.samples} samples = "
          f"{args.pixels ** 2 * args.samples / 1e6:.0f}M kernel evaluations")

    t0 = time.time()
    horizon = render(cube, cam_horizon, n_samples=args.samples, tile=args.tile)
    print(f"render horizon-referenced   {time.time() - t0:6.1f}s")

    visible = north.total > 0
    pol_ratio = north.polarised[visible] / north.total[visible]
    print()
    print(f"degree of polarisation      {pol_ratio.min():.3f} .. {pol_ratio.max():.3f} "
          f"(median {np.median(pol_ratio):.3f})")
    print(f"brightness dynamic range    {north.total[visible].max() / north.total[visible].min():.3g}")
    print(f"solar north from zenith     {solar_north_from_zenith().to_value(u.deg):.2f} deg")

    args.outdir.mkdir(parents=True, exist_ok=True)
    hero = plot_corona(
        horizon,
        args.outdir / "corona_prediction.png",
        title="The corona of 12 August 2026, as it should appear from Colmenar Viejo",
        subtitle=f"ADAPT-GONG {adapt.map_time.isot[:16]}Z   ·   PFSS R$_{{ss}}$ = "
                 f"{args.rss} R☉   ·   topology-informed density   ·   "
                 f"Thomson scattering, {args.samples} samples/ray",
        annotate_directions=("zenith", "horizon →"),
        figsize=9.0,
    )
    channels = plot_channels(
        north, horizon, args.outdir / "corona_channels.png",
        map_time=f"ADAPT-GONG {adapt.map_time.isot}Z  ·  realisation "
                 f"{args.realisation}  ·  Rss = {args.rss} R☉",
    )

    manifest = args.outdir / "corona_prediction.manifest.json"
    manifest.write_text(json.dumps({
        "map_file": adapt.path.name,
        "map_time_utc": adapt.map_time.isot,
        "realisation": args.realisation,
        "rss": args.rss,
        "pixels": args.pixels,
        "samples_per_ray": args.samples,
        "field_of_view_rsun": args.fov,
        "density_cube": cube.summary(),
        "observer": geom.summary(),
        "solar_north_from_zenith_deg": float(
            solar_north_from_zenith().to_value(u.deg)
        ),
        "median_degree_of_polarisation": float(np.median(pol_ratio)),
    }, indent=2, default=str) + "\n")

    print()
    print(f"hero     : {hero}")
    print(f"channels : {channels}")
    print(f"manifest : {manifest}")


if __name__ == "__main__":
    main()
