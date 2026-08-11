"""Figure 4: the corona itself.

The point of the whole pipeline. Two things make the difference between a
white blob and something that looks like an eclipse:

* **the radial filter**, because brightness falls by 3.3 orders of magnitude
  across the frame and no display has that range;
* **the colour**, which is not decoration. The K-corona is Thomson-scattered
  photospheric light, so it is very nearly the colour of the photosphere:
  white, with the faintest warm cast. Rendering it in a false-colour palette
  would be prettier and less true.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from corona26.radiation.camera import Camera
from corona26.radiation.render import Render, radial_filter

# Black through a warm ember to photospheric white. The warm cast in the inner
# corona is a real, if subtle, feature of eclipse photographs.
CORONA_CMAP = LinearSegmentedColormap.from_list(
    "corona",
    [
        (0.00, "#000000"),
        (0.08, "#120e0a"),
        (0.22, "#3b332a"),
        (0.40, "#6e6355"),
        (0.58, "#9d9082"),
        (0.74, "#c6bcae"),
        (0.88, "#e6ded2"),
        (1.00, "#fffdf8"),
    ],
)


# Display pipeline, in three steps that each do one job:
#
#   1. divide out the measured radial profile completely, so structure has the
#      same amplitude at every height and streamers are visible at 3 R_sun as
#      well as at 1.1;
#   2. multiply back a gentle synthetic falloff, so the image still reads as
#      something streaming away from a star rather than a flat disc;
#   3. gamma-compress what is left.
#
# Only step 1 has any claim to being principled. Steps 2 and 3 are display
# choices, tuned once and then applied identically to every image in a
# comparison — which is the only property that actually matters, because
# comparing differently-processed images is how you produce a meaningless
# figure.
FILTER_STRENGTH = 1.0
COSMETIC_FALLOFF = 2.0
DISPLAY_GAMMA = 0.78


def stretch(
    image: np.ndarray,
    camera: Camera,
    *,
    falloff: float = COSMETIC_FALLOFF,
    gamma: float = DISPLAY_GAMMA,
    clip: float = 99.7,
):
    """Turn a radially flattened image into display values in [0, 1]."""
    x, y = camera.image_axes()
    xx, yy = np.meshgrid(x, y)
    radius = np.hypot(xx, yy)

    shaped = np.where(image > 0, image * np.clip(radius, 1.0, None) ** -falloff, 0.0)

    visible = shaped[shaped > 0]
    if visible.size == 0:
        return np.zeros_like(shaped)
    hi = np.percentile(visible, clip)
    lo = 0.0
    out = np.clip((shaped - lo) / max(hi - lo, 1e-30), 0.0, 1.0)
    return np.where(shaped > 0, out**gamma, 0.0)


def _draw(ax, image, camera: Camera, *, title: str = "", subtitle: str = ""):
    half = camera.field_of_view
    ax.imshow(
        image, origin="lower", extent=[-half, half, -half, half],
        cmap=CORONA_CMAP, vmin=0.0, vmax=1.0, interpolation="bilinear",
    )
    # The occulting body. Drawn explicitly rather than left as zeros so the
    # edge is clean at any resolution.
    ax.add_patch(plt.Circle((0, 0), 1.0, color="#000000", zorder=3))
    ax.add_patch(
        plt.Circle((0, 0), 1.0, fill=False, color="#3a2a1c", lw=0.8, zorder=4)
    )
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, color="#e8e4dc", fontsize=12, pad=6)
    if subtitle:
        ax.text(
            0.5, -0.035, subtitle, transform=ax.transAxes, ha="center", va="top",
            color="#8d857a", fontsize=8.5,
        )


def plot_corona(
    render_result: Render,
    outpath: str | Path,
    *,
    title: str = "",
    subtitle: str = "",
    annotate_directions: tuple[str, str] | None = None,
    figsize: float = 8.0,
) -> Path:
    """A single hero image of the predicted corona."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    camera = render_result.camera
    filtered = radial_filter(render_result.total, camera, strength=FILTER_STRENGTH)
    image = stretch(filtered, camera)

    fig, ax = plt.subplots(figsize=(figsize, figsize * 1.06))
    fig.patch.set_facecolor("black")
    _draw(ax, image, camera, title=title)

    if annotate_directions:
        up_label, right_label = annotate_directions
        style = dict(color="#7d7469", fontsize=9, ha="center", va="center")
        ax.text(0, camera.field_of_view * 0.94, up_label, **style)
        ax.text(camera.field_of_view * 0.88, 0, right_label, **style)

    if subtitle:
        # Figure-level so long provenance strings wrap inside the canvas
        # instead of running off the edge.
        fig.text(
            0.5, 0.022, subtitle, ha="center", va="bottom",
            color="#8d857a", fontsize=8.5, wrap=True,
        )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.06)
    fig.savefig(outpath, dpi=170, facecolor="black")
    plt.close(fig)
    return outpath


def plot_channels(
    north_up: Render,
    horizon: Render,
    outpath: str | Path,
    *,
    map_time: str = "",
) -> Path:
    """Three panels: total brightness, polarised brightness, and the real view."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0))
    fig.patch.set_facecolor("black")

    _draw(
        axes[0], stretch(radial_filter(north_up.total, north_up.camera, strength=FILTER_STRENGTH), north_up.camera),
        north_up.camera,
        title="Total brightness $B$",
        subtitle="solar north up — comparable with published predictions",
    )
    _draw(
        axes[1], stretch(radial_filter(north_up.polarised, north_up.camera, strength=FILTER_STRENGTH), north_up.camera),
        north_up.camera,
        title="Polarised brightness $pB$",
        subtitle="isolates the K-corona; the dust corona is nearly unpolarised",
    )
    _draw(
        axes[2], stretch(radial_filter(horizon.total, horizon.camera, strength=FILTER_STRENGTH), horizon.camera),
        horizon.camera,
        title="As seen from Colmenar Viejo",
        subtitle="zenith up — solar north tilted 34.7° from vertical",
    )

    fig.suptitle(
        f"corona26 — predicted K-corona for 2026-08-12 20:31 CEST\n{map_time}",
        color="#e8e4dc", fontsize=12.5, y=0.99,
    )
    fig.tight_layout(pad=0.7, rect=(0, 0, 1, 0.94))
    fig.savefig(outpath, dpi=160, facecolor="black")
    plt.close(fig)
    return outpath
