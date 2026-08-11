"""Figure 3: coronal holes, and how much of the Sun is open.

Open field regions are where the solar wind escapes. They are underdense, and
so they are the *dark* parts of an eclipse corona. Their footpoint map is the
main input to the electron-density proxy, which is why this figure comes
before any brightness is computed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from corona26.magnetic.trace import Topology

EXTENT = [0.0, 360.0, -90.0, 90.0]


def _sin_lat_axis(n_lat: int) -> np.ndarray:
    edges = np.linspace(-1, 1, n_lat + 1)
    return np.degrees(np.arcsin(0.5 * (edges[:-1] + edges[1:])))


def open_field_image(topology: Topology) -> np.ndarray:
    """Encode topology as -1 (open negative), 0 (closed), +1 (open positive)."""
    img = np.zeros(topology.is_open.shape, dtype=float)
    img[topology.is_open] = topology.polarity[topology.is_open]
    return img


def plot_topology(
    topology: Topology,
    open_fractions: dict[float, list[float]],
    outpath: str | Path,
    *,
    rss: float = 2.5,
    map_time: str = "",
    limbs: dict[str, float] | None = None,
) -> Path:
    """Coronal-hole map plus open area fraction across the ensemble."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 8.4),
        gridspec_kw={"height_ratios": [4.4, 3.2]}, constrained_layout=True,
    )

    ax = axes[0]
    cmap = ListedColormap(["#2a6f97", "#f2f2f2", "#d1495b"])
    ax.imshow(
        open_field_image(topology), origin="lower", extent=EXTENT,
        cmap=cmap, vmin=-1.5, vmax=1.5, aspect="auto", interpolation="nearest",
    )
    ax.set_xlabel("Carrington longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    ax.set_title(
        f"Open field at the photosphere — coronal holes at $R_{{ss}} = {rss}$ "
        f"$R_\\odot$\n{map_time}   ·   "
        f"{100 * topology.open_area_fraction:.1f}% of the surface is open",
        fontsize=11,
    )
    ax.axhline(0, color="#444", lw=0.4, alpha=0.4)

    if limbs:
        for name, lon in limbs.items():
            solid = name != "disk centre"
            ax.axvline(lon, color="#111", ls="-" if solid else "--", lw=1.1,
                       alpha=0.75)
            ax.annotate(
                name, (lon, 86), fontsize=8, ha="center", va="top", color="#111",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2),
            )

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#d1495b", ec="none"),
        plt.Rectangle((0, 0), 1, 1, fc="#2a6f97", ec="none"),
        plt.Rectangle((0, 0), 1, 1, fc="#f2f2f2", ec="#bbb"),
    ]
    ax.legend(
        handles, ["open, positive", "open, negative", "closed (streamers)"],
        loc="lower center", bbox_to_anchor=(0.5, -0.29), ncol=3,
        frameon=False, fontsize=9,
    )

    ax = axes[1]
    radii = sorted(open_fractions)
    means = np.array([100 * np.mean(open_fractions[r]) for r in radii])
    lo = np.array([100 * np.min(open_fractions[r]) for r in radii])
    hi = np.array([100 * np.max(open_fractions[r]) for r in radii])

    ax.fill_between(radii, lo, hi, color="#2a6f97", alpha=0.2,
                    label="spread across the 12 ADAPT realisations")
    ax.plot(radii, means, "o-", color="#2a6f97", lw=1.8, ms=6,
            label="ensemble mean")
    ax.axvline(2.5, color="#d1495b", ls="--", lw=1.2)
    ax.annotate("conventional 2.5 $R_\\odot$", (2.5, means.max()),
                color="#d1495b", fontsize=9, ha="center", va="bottom")
    ax.set_xlabel("source surface radius $R_{ss}$ [$R_\\odot$]")
    ax.set_ylabel("open surface area [%]")
    ax.set_title(
        "How much of the Sun is open — and how much that depends on a "
        "parameter nobody has measured",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15)

    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath
