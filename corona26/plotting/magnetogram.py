"""Figure 1: the boundary condition, and how much we disagree about it."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

from corona26.data.adapt import AdaptMap, flux_balance, latitude_centres


def _extent(shape: tuple[int, int]) -> list[float]:
    return [0.0, 360.0, -90.0, 90.0]


def spread_vs_longitude(adapt: AdaptMap) -> np.ndarray:
    """Area-weighted mean ensemble spread as a function of Carrington longitude.

    This is effectively a map of *how long ago we last looked*. Longitudes on
    the currently visible hemisphere are pinned by recent GONG observations
    and the realisations agree; longitudes that have spent longest on the far
    side are pure flux-transport extrapolation and the realisations diverge.
    """
    if adapt.stddev is None:
        raise ValueError("this ADAPT file carries no STDDEV extension")
    lat = np.deg2rad(latitude_centres(adapt.stddev.shape[0]))
    w = np.cos(lat)[:, None] * np.ones_like(adapt.stddev)
    valid = np.isfinite(adapt.stddev)
    w = np.where(valid, w, 0.0)
    values = np.where(valid, adapt.stddev, 0.0)
    return np.sum(values * w, axis=0) / np.sum(w, axis=0)


def limb_longitudes(l0_deg: float) -> dict[str, float]:
    """Carrington longitudes of disk centre and limbs at a given L0.

    The Carrington longitude of the central meridian *decreases* with time, so
    features reach central meridian from lower longitudes: the east limb —
    the one that has most recently rotated out of the far side — is at
    ``L0 - 90``, not ``L0 + 90``. Getting this backwards inverts the entire
    uncertainty argument.
    """
    return {
        "east limb": (l0_deg - 90) % 360,
        "disk centre": l0_deg % 360,
        "west limb": (l0_deg + 90) % 360,
    }


def plot_magnetogram(
    adapt: AdaptMap,
    outpath: str | Path,
    *,
    realisation: int = 0,
    linthresh: float = 5.0,
    vmax: float | None = None,
    l0_at_totality: float | None = None,
) -> Path:
    """Plot one realisation next to the ensemble spread.

    Symmetric-log colour scaling is not decoration. Photospheric fields span
    quiet-Sun tenths of a gauss to active-region hundreds; on a linear scale
    the quiet Sun — which is most of the surface, and which anchors most of
    the open flux — renders as flat grey.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    br = adapt.data[realisation]
    spread = adapt.stddev
    lat = latitude_centres(br.shape[0])

    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(br), 99.9))

    has_spread = spread is not None
    n_panels = 3 if has_spread else 1
    heights = [4.2, 4.2, 2.4] if has_spread else [4.2]
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(11, sum(heights)),
        gridspec_kw={"height_ratios": heights}, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)
    im = axes[0].imshow(
        br, origin="lower", extent=_extent(br.shape), cmap="RdBu_r", norm=norm,
        aspect="auto",
    )
    fig.colorbar(im, ax=axes[0], label="$B_r$ [G]", extend="both")
    axes[0].set_title(
        f"ADAPT-GONG $B_r$ — realisation {realisation} of {adapt.n_realisations}"
        f"\n{adapt.map_time.isot}Z   ·   CM Carrington longitude "
        f"{adapt.carrington_longitude.value:.1f}°",
        fontsize=11,
    )

    if spread is not None:
        im2 = axes[1].imshow(
            spread, origin="lower", extent=_extent(spread.shape),
            cmap="magma", aspect="auto",
            vmin=0, vmax=float(np.nanpercentile(spread, 99.5)),
        )
        fig.colorbar(im2, ax=axes[1], label="ensemble $\\sigma$ [G]", extend="max")
        axes[1].set_title(
            "Spread across the 12 realisations — where the flux-transport model "
            "disagrees with itself",
            fontsize=11,
        )

    for ax in axes[: 2 if has_spread else 1]:
        ax.set_xlabel("Carrington longitude [deg]")
        ax.set_ylabel("latitude [deg]")
        ax.axhline(0, color="k", lw=0.4, alpha=0.3)

    if has_spread:
        ax = axes[2]
        profile = spread_vs_longitude(adapt)
        lon = np.arange(0.5, 360.0, 360.0 / profile.size)
        ax.plot(lon, profile, color="#b3306e", lw=1.6)
        ax.fill_between(lon, 0, profile, color="#b3306e", alpha=0.15)
        ax.set_xlim(0, 360)
        ax.set_ylim(0, profile.max() * 1.25)
        ax.set_xlabel("Carrington longitude [deg]")
        ax.set_ylabel("mean $\\sigma$ [G]")
        ax.set_title(
            "Boundary-condition uncertainty vs longitude — a map of how long "
            "ago we last looked",
            fontsize=11,
        )
        if l0_at_totality is not None:
            styles = {
                "east limb": ("#d1495b", "-"),
                "disk centre": ("#2a6f97", "--"),
                "west limb": ("#d1495b", "-"),
            }
            for name, l in limb_longitudes(l0_at_totality).items():
                colour, ls = styles[name]
                ax.axvline(l, color=colour, ls=ls, lw=1.2, alpha=0.9)
                ax.annotate(
                    name, (l, profile.max() * 1.14), color=colour, fontsize=8,
                    ha="center", va="center",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2),
                )

    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def ensemble_summary(adapt: AdaptMap) -> dict[str, float]:
    """Flux diagnostics across all realisations."""
    balances = [flux_balance(adapt.data[i]) for i in range(adapt.n_realisations)]
    ratios = np.array([b["monopole_ratio"] for b in balances])
    unsigned = np.array([b["unsigned_flux"] for b in balances])
    return {
        "monopole_ratio_min": float(ratios.min()),
        "monopole_ratio_max": float(ratios.max()),
        "monopole_ratio_mean": float(ratios.mean()),
        "unsigned_flux_mean": float(unsigned.mean()),
        "unsigned_flux_spread_percent": float(
            100 * (unsigned.max() - unsigned.min()) / unsigned.mean()
        ),
    }
