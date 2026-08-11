"""Single-Rss source-surface panels, for the interactive explorer on the site.

One image per source-surface radius so the page can swap between them and show
that the parameter changes the topology of the streamer belt, not just its
amplitude.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

from corona26.plotting.pfss_ensemble import EXTENT, neutral_line_latitude


def polarity_reversals(source_surface_br: np.ndarray) -> float:
    """Mean number of polarity reversals per longitude.

    A single clean streamer belt gives ~1. Values above 2 mean the belt is
    multi-branched and the eclipse would show several distinct streamers.
    """
    data = np.asarray(source_surface_br, dtype=np.float64)
    return float(
        np.mean([np.sum(np.diff(np.signbit(data[:, j])) != 0)
                 for j in range(data.shape[1])])
    )


def plot_single_rss(
    source_surface_br: np.ndarray,
    rss: float,
    outpath: str | Path,
    *,
    open_flux_value: float | None = None,
    dark: bool = True,
) -> Path:
    """One source-surface panel with its neutral line and headline numbers."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    ink = "#c0caf5" if dark else "#1f2335"
    paper = "#16161e" if dark else "#ffffff"
    subtle = "#565f89" if dark else "#6c7086"

    data = np.asarray(source_surface_br, dtype=np.float64)
    vmax = float(np.nanpercentile(np.abs(data), 99.5))

    fig, ax = plt.subplots(figsize=(10, 3.4), constrained_layout=True)
    fig.patch.set_facecolor(paper)
    ax.set_facecolor(paper)

    ax.imshow(
        data, origin="lower", extent=EXTENT, cmap="RdBu_r", aspect="auto",
        norm=SymLogNorm(linthresh=max(vmax / 50, 1e-3), vmin=-vmax, vmax=vmax,
                        base=10),
    )
    nl = neutral_line_latitude(data)
    lon = np.linspace(0, 360, nl.size)
    ax.plot(lon, nl, color="#16161e", lw=2.6)
    ax.plot(lon, nl, color="#e0af68", lw=1.3)

    reversals = polarity_reversals(data)
    label = f"$R_{{ss}} = {rss}\\,R_\\odot$   ·   {reversals:.2f} polarity reversals per longitude"
    if open_flux_value is not None:
        label += f"   ·   open flux {open_flux_value:.1f} G$R_\\odot^2$"
    ax.set_title(label, fontsize=11, color=ink, pad=8)

    ax.set_xlabel("Carrington longitude [deg]", fontsize=9, color=subtle)
    ax.set_ylabel("latitude [deg]", fontsize=9, color=subtle)
    ax.set_ylim(-90, 90)
    ax.tick_params(colors=subtle, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(subtle)
        spine.set_linewidth(0.6)

    fig.savefig(outpath, dpi=140, facecolor=paper)
    plt.close(fig)
    return outpath
