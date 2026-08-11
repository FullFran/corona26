"""Figure 2: what the source-surface choice actually changes.

The neutral line at the source surface — where ``Br`` changes sign — is the
base of the heliospheric current sheet, and the streamer belt sits on it. It
is the single most consequential structure for an eclipse image, because
streamers seen edge-on at the limb are what the eye actually registers during
totality.

So the honest question is not "what does the corona look like" but "how much
does the streamer belt move when we vary a parameter nobody has measured".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

EXTENT = [0.0, 360.0, -90.0, 90.0]


def _sin_lat_axis(n_lat: int) -> np.ndarray:
    edges = np.linspace(-1, 1, n_lat + 1)
    return np.degrees(np.arcsin(0.5 * (edges[:-1] + edges[1:])))


def neutral_line_latitude(source_surface_br: np.ndarray) -> np.ndarray:
    """Latitude of the polarity inversion, per longitude.

    Interpolates the zero crossing nearest the equator. Where the field is
    multipolar there can be several; we take the one closest to the equator,
    which is the one that carries the main streamer belt.
    """
    data = np.asarray(source_surface_br, dtype=np.float64)
    lat = _sin_lat_axis(data.shape[0])
    out = np.full(data.shape[1], np.nan)

    for j in range(data.shape[1]):
        column = data[:, j]
        sign_change = np.where(np.diff(np.signbit(column)))[0]
        if sign_change.size == 0:
            continue
        crossings = []
        for i in sign_change:
            b0, b1 = column[i], column[i + 1]
            if b1 == b0:
                continue
            frac = -b0 / (b1 - b0)
            crossings.append(lat[i] + frac * (lat[i + 1] - lat[i]))
        if crossings:
            out[j] = min(crossings, key=abs)
    return out


def plot_rss_ensemble(
    solutions: dict[float, np.ndarray],
    open_fluxes: dict[float, list[float]],
    outpath: str | Path,
    *,
    map_time: str = "",
) -> Path:
    """Source-surface field and neutral line for each Rss, plus open flux spread."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    radii = sorted(solutions)
    n = len(radii)
    fig = plt.figure(figsize=(12, 2.1 * n + 3.4), constrained_layout=True)
    gs = fig.add_gridspec(n + 1, 1, height_ratios=[2.1] * n + [3.2])

    colours = plt.cm.viridis(np.linspace(0.1, 0.85, n))

    for k, rss in enumerate(radii):
        ax = fig.add_subplot(gs[k])
        data = solutions[rss]
        vmax = float(np.nanpercentile(np.abs(data), 99.5))
        ax.imshow(
            data, origin="lower", extent=EXTENT, cmap="RdBu_r", aspect="auto",
            norm=SymLogNorm(linthresh=max(vmax / 50, 1e-3), vmin=-vmax, vmax=vmax,
                            base=10),
        )
        nl = neutral_line_latitude(data)
        ax.plot(np.linspace(0, 360, nl.size), nl, color="#111", lw=1.8)
        ax.plot(np.linspace(0, 360, nl.size), nl, color="#f5d90a", lw=0.9)
        ax.set_ylabel("lat [deg]")
        ax.set_ylim(-90, 90)
        ax.text(
            0.006, 0.93, f"$R_{{ss}} = {rss}\\,R_\\odot$", transform=ax.transAxes,
            va="top", fontsize=10, fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=2),
        )
        if k < n - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Carrington longitude [deg]")
        if k == 0:
            ax.set_title(
                "Source-surface $B_r$ and the neutral line — the streamer belt, "
                f"as a function of a parameter nobody has measured\n{map_time}",
                fontsize=11,
            )

    ax = fig.add_subplot(gs[n])
    means, lo, hi = [], [], []
    for rss in radii:
        f = np.array(open_fluxes[rss])
        means.append(f.mean())
        lo.append(f.min())
        hi.append(f.max())
    means, lo, hi = np.array(means), np.array(lo), np.array(hi)

    ax.fill_between(radii, lo, hi, color="#2a6f97", alpha=0.2,
                    label="spread across the 12 ADAPT realisations")
    ax.plot(radii, means, "o-", color="#2a6f97", lw=1.8, ms=6,
            label="ensemble mean")
    for x, y, c in zip(radii, means, colours):
        ax.plot([x], [y], "o", color=c, ms=6)
    ax.axvline(2.5, color="#d1495b", ls="--", lw=1.2)
    ax.annotate(
        "the conventional 2.5 $R_\\odot$", (2.5, ax.get_ylim()[1]),
        color="#d1495b", fontsize=9, ha="center", va="top",
        xytext=(0, -6), textcoords="offset points",
    )
    ax.set_xlabel("source surface radius $R_{ss}$ [$R_\\odot$]")
    ax.set_ylabel("open flux [G $R_\\odot^2$]")
    ax.set_title(
        "Open flux vs $R_{ss}$. The band is the 12-realisation spread — barely "
        "visible, because\nthe boundary uncertainty is small next to the choice "
        "of source surface",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15)

    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath
