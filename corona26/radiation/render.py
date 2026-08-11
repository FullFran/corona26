"""The renderer: line-of-sight integration of Thomson-scattered light.

For every pixel,

    I(x, y) = integral over the line of sight of  ne(r) * K(r, chi)  dl

with ``K`` the van de Hulst kernel. There is no extinction term because the
corona is optically thin, which is the whole reason this is a single forward
pass instead of a radiative-transfer solve.

Every sample is independent of every other, so this is a pure map-reduce and
the natural place for a GPU. It is written against NumPy first as the
reference implementation, and rendered in tiles so memory stays bounded no
matter the resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corona26.plasma.topology_density import DensityCube, sample_density
from corona26.radiation.camera import (
    Camera,
    cartesian_to_heliographic,
    scattering_cosine,
)
from corona26.radiation.thomson import scattering_intensities


@dataclass(frozen=True)
class Render:
    """A rendered corona in both observable channels."""

    total: np.ndarray       # B  = I_tan + I_rad
    polarised: np.ndarray   # pB = I_tan - I_rad
    camera: Camera
    n_samples: int

    @property
    def occulted(self) -> np.ndarray:
        """Mask of the lunar disk — pixels inside 1 solar radius."""
        x, y = self.camera.image_axes()
        xx, yy = np.meshgrid(x, y)
        return np.hypot(xx, yy) < 1.0


def _stretched_samples(half: np.ndarray, n_samples: int, stretch: float):
    """Quadrature points along each ray, concentrated near closest approach.

    Uniform spacing is badly matched to this integrand. Density falls as
    steeply as ``r^-16`` near the surface, so almost all the signal comes from
    a short stretch of the ray around its closest approach to the Sun, while
    uniform sampling spends most of its points far out where nothing happens.

    Mapping ``z = half * sinh(a t) / sinh(a)`` for ``t`` uniform on [-1, 1]
    packs samples near ``z = 0`` and stretches them in the tails, turning a
    first-order convergence into something far cheaper for the same accuracy.
    """
    t = np.linspace(-1.0, 1.0, n_samples)
    if stretch <= 0:
        z_unit = t
        weight = np.full_like(t, 2.0 / (n_samples - 1))
    else:
        z_unit = np.sinh(stretch * t) / np.sinh(stretch)
        # dz/dt for the change of variables, times the uniform dt.
        weight = (
            stretch * np.cosh(stretch * t) / np.sinh(stretch)
            * (2.0 / (n_samples - 1))
        )
    # Trapezoid: half weight at the two endpoints.
    weight = weight.copy()
    weight[0] *= 0.5
    weight[-1] *= 0.5

    zz = half[..., None] * z_unit
    dl = half[..., None] * weight
    return zz, dl


def render(
    cube: DensityCube,
    camera: Camera,
    *,
    n_samples: int = 384,
    tile: int = 64,
    u_limb: float = 0.6,
    stretch: float = 3.0,
) -> Render:
    """Integrate the corona along every line of sight.

    Parameters
    ----------
    n_samples
        Quadrature points per ray. Convergence is checked by doubling, not
        assumed — see `quadrature_error`.
    tile
        Pixels per tile. Peak memory is ``tile^2 * n_samples`` floats, so this
        is the knob that lets a 4096-pixel render run in bounded VRAM.
    """
    x_axis, y_axis = camera.image_axes()
    image_total = np.zeros((camera.n_pixels, camera.n_pixels))
    image_pol = np.zeros_like(image_total)

    # The integration runs from in front of the Sun to behind it. Beyond the
    # cube's outer radius the density is zero, so the finite span loses
    # nothing while keeping every sample useful.
    span = np.sqrt(max(cube.r_max**2 - 0.0, 0.0))

    for iy in range(0, camera.n_pixels, tile):
        for ix in range(0, camera.n_pixels, tile):
            us = x_axis[ix : ix + tile]
            vs = y_axis[iy : iy + tile]
            uu, vv = np.meshgrid(us, vs)
            # Display coordinates first, then rotate into the solar frame, so
            # the roll costs nothing and introduces no resampling blur.
            xx, yy = camera.display_to_solar(uu, vv)

            impact = np.hypot(xx, yy)
            # Rays with impact parameter beyond the cube never enter it, and
            # rays inside 1 solar radius are blocked by the Moon. Leaving the
            # occulted disk unmasked is not merely cosmetic: those pixels
            # integrate straight through the densest plasma in the model, so
            # they dominate any error or brightness statistic taken over the
            # whole frame while corresponding to nothing an observer can see.
            active = (impact < cube.r_max) & (impact >= 1.0)

            # Half-chord through the outer sphere, per ray.
            half = np.sqrt(np.clip(cube.r_max**2 - impact**2, 0.0, None))
            zz, dl = _stretched_samples(half, n_samples, stretch)

            x3 = xx[..., None]
            y3 = yy[..., None]

            r, sin_lat, lon = cartesian_to_heliographic(
                x3, y3, zz, b0_deg=camera.b0_deg, l0_deg=camera.l0_deg
            )
            cos_chi = scattering_cosine(x3, y3, zz, camera.observer_distance)

            ne = sample_density(cube, r, sin_lat, lon)
            del sin_lat, lon

            # The kernel is undefined below the photosphere; those samples are
            # already zero-weighted by the density but must not produce NaN.
            r_safe = np.clip(r, 1.0 + 1e-9, None)
            del r

            # One pass, both channels. Calling total_brightness and
            # polarised_brightness separately would evaluate the van de Hulst
            # coefficients twice over the largest arrays in the program.
            i_tan, i_rad = scattering_intensities(r_safe, cos_chi, u=u_limb)
            del r_safe, cos_chi

            weight = ne * dl
            del ne
            block_total = np.sum(weight * (i_tan + i_rad), axis=-1)
            block_pol = np.sum(weight * (i_tan - i_rad), axis=-1)
            del weight, i_tan, i_rad

            image_total[iy : iy + tile, ix : ix + tile] = np.where(
                active, block_total, 0.0
            )
            image_pol[iy : iy + tile, ix : ix + tile] = np.where(
                active, block_pol, 0.0
            )

    return Render(
        total=image_total, polarised=image_pol, camera=camera, n_samples=n_samples
    )


def quadrature_error(cube: DensityCube, camera: Camera, *, n_samples: int) -> float:
    """Relative change in the image when the sample count is doubled.

    The honest way to choose ``n_samples``: measure, do not guess.
    """
    coarse = render(cube, camera, n_samples=n_samples).total
    fine = render(cube, camera, n_samples=2 * n_samples).total
    mask = fine > 0
    return float(
        np.max(np.abs(coarse[mask] - fine[mask]) / fine[mask])
    )


def radial_filter(image: np.ndarray, camera: Camera, *, strength: float = 1.0):
    """Divide out the steep radial falloff so structure is visible at all.

    Coronal brightness drops about four orders of magnitude between 1.1 and 3
    solar radii. Without this every eclipse image is a white blob with a black
    background, which is why every published prediction applies some version of
    it. Applied identically to every image we compare, or the comparison is
    meaningless.
    """
    x, y = camera.image_axes()
    xx, yy = np.meshgrid(x, y)
    radius = np.hypot(xx, yy)

    bins = np.linspace(1.0, camera.field_of_view * np.sqrt(2), 220)
    idx = np.digitize(radius, bins)
    profile = np.ones_like(bins, dtype=np.float64)

    for i in range(1, bins.size):
        sel = (idx == i) & (image > 0)
        if np.any(sel):
            profile[i] = np.median(image[sel])

    # Fill empty bins so the divisor stays monotone and smooth.
    good = profile > 0
    profile = np.interp(np.arange(bins.size), np.flatnonzero(good), profile[good])

    divisor = np.interp(radius, bins, profile)
    out = np.where(divisor > 0, image / divisor**strength, 0.0)
    return np.where(radius >= 1.0, out, 0.0)
