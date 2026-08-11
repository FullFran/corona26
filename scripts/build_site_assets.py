"""Render the per-Rss panels and htmx fragments the site swaps between.

    uv run python scripts/build_site_assets.py --file data/raw/<adapt>.fts.gz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from corona26.data.adapt import load_adapt, realisation_map, to_cea
from corona26.magnetic.pfss import RSS_ENSEMBLE, open_flux, solve
from corona26.plotting.rss_panels import plot_single_rss, polarity_reversals

FRAGMENT = """<figure class="sim__figure">
  <img src="assets/rss/ss_{slug}.png"
       alt="Source-surface radial field and neutral line for a source surface at {rss} solar radii"
       width="1400" height="476">
  <figcaption data-es="{caption_es}">
    <strong>R<sub>ss</sub> = {rss} R<sub>&#9737;</sub></strong> —
    {reversals} polarity reversals per longitude, open flux {flux} G&nbsp;R<sub>&#9737;</sub><sup>2</sup>.
    {commentary}
  </figcaption>
</figure>
"""

CAPTION_ES = (
    "<strong>R<sub>ss</sub> = {rss} R<sub>&#9737;</sub></strong> — "
    "{reversals} inversiones de polaridad por longitud, flujo abierto "
    "{flux} G&nbsp;R<sub>&#9737;</sub><sup>2</sup>. {commentary}"
)

COMMENTARY_ES = {
    1.3: "La banda es multi-ramificada: varios streamers distintos y una línea "
         "neutra que serpentea más de 50&deg; en latitud. Predice una corona "
         "cargada de estructura.",
    1.5: "Sigue siendo multi-ramificada, pero las regiones cerradas pequeñas "
         "empiezan a fundirse con la banda principal.",
    2.0: "Se acerca a una banda única. La mayoría de líneas neutras secundarias "
         "se han cerrado por debajo de la source surface.",
    2.5: "La elección convencional desde Altschuler y Newkirk (1969). Una banda "
         "dominante con un par de excursiones.",
    3.0: "Una única banda limpia de dos sectores. Casi toda la estructura ha "
         "quedado confinada bajo la source surface: una corona simple y "
         "tranquila.",
}

COMMENTARY = {
    1.3: "The belt is multi-branched: several distinct streamers, and a neutral "
         "line that wanders more than 50&deg; in latitude. This predicts a busy, "
         "structured corona.",
    1.5: "Still multi-branched, but the smaller closed regions are starting to "
         "merge into the main belt.",
    2.0: "Approaching a single belt. Most of the secondary neutral lines have "
         "closed off below the source surface.",
    2.5: "The conventional choice since Altschuler &amp; Newkirk (1969). One "
         "dominant belt with a couple of excursions.",
    3.0: "A single clean two-sector belt. Nearly all structure has been "
         "confined below the source surface — a simple, quiet-looking corona.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--nr", type=int, default=100)
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args()

    adapt = load_adapt(args.file)
    cea = to_cea(realisation_map(adapt, 0))

    assets = args.docs / "assets" / "rss"
    fragments = args.docs / "fragments"
    assets.mkdir(parents=True, exist_ok=True)
    fragments.mkdir(parents=True, exist_ok=True)

    summary = {}
    for rss in RSS_ENSEMBLE:
        sol = solve(cea, rss=rss, nr=args.nr)
        ss = np.asarray(sol.source_surface_br.data, dtype=np.float64)
        flux = open_flux(sol)
        rev = polarity_reversals(ss)
        slug = str(rss).replace(".", "_")

        plot_single_rss(ss, rss, assets / f"ss_{slug}.png", open_flux_value=flux)
        caption_es = CAPTION_ES.format(
            rss=rss, reversals=f"{rev:.2f}".replace(".", ","),
            flux=f"{flux:.1f}".replace(".", ","), commentary=COMMENTARY_ES[rss],
        ).replace('"', "&quot;")
        (fragments / f"rss-{slug}.html").write_text(
            FRAGMENT.format(
                slug=slug, rss=rss, reversals=f"{rev:.2f}", flux=f"{flux:.1f}",
                commentary=COMMENTARY[rss], caption_es=caption_es,
            )
        )
        summary[str(rss)] = {"open_flux": flux, "reversals": rev}
        print(f"  Rss={rss:4.1f}  reversals={rev:4.2f}  flux={flux:6.2f}")

    (args.docs / "assets" / "rss-summary.json").write_text(
        json.dumps(
            {
                "map_time_utc": adapt.map_time.isot,
                "map_file": adapt.path.name,
                "realisation": 0,
                "nr": args.nr,
                "values": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nassets   : {assets}")
    print(f"fragments: {fragments}")


if __name__ == "__main__":
    main()
