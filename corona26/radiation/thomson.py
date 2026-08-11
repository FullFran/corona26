"""Thomson scattering: the one part of this pipeline that is exact.

A free electron in the corona is accelerated by the electric field of
photospheric light and re-radiates. Classical, elastic, wavelength-independent.
The corona is optically thin — ``tau`` of order 1e-6 — so single scattering is
not an approximation but an excellent description of reality. There is no
``exp(-tau)`` term anywhere below, and that absence is why a whole eclipse image
is one line-of-sight quadrature rather than a Monte Carlo.

The subtlety is that **the Sun is not a point source**. An electron at 1.5 solar
radii sees a disk 42 degrees wide, so the scattering angle varies across the
disk and must be integrated over it, with limb darkening. Doing that integral
analytically is what van de Hulst (1950) and Billings (1966) give us: four
functions of the disk half-angle alone, in closed form. One evaluation per
sample point, no nested integral at runtime — which is exactly what makes a
GPU render of hundreds of millions of samples tractable.

Treating the Sun as a point source is the classic error here, and it gets both
the brightness and the polarisation wrong.

References
----------
van de Hulst (1950); Billings (1966), *A Guide to the Solar Corona*;
Inhester (2015), arXiv:1512.00651.
"""

from __future__ import annotations

import numpy as np

# Classical electron radius [m] and the Thomson cross-section it implies.
R_ELECTRON = 2.8179403262e-15
SIGMA_THOMSON = (8.0 * np.pi / 3.0) * R_ELECTRON**2  # 6.652e-29 m^2

# Visible-light limb darkening coefficient. Billings uses 0.6 at 5000 A.
LIMB_DARKENING_U = 0.6


def _omega_terms(r):
    """Geometry of the solar disk as seen from a scattering electron.

    ``sin(omega) = 1 / r`` with ``r`` in solar radii, so an electron close to
    the surface sees a wide disk and one far away sees nearly a point.
    """
    r = np.asarray(r, dtype=np.float64)
    if np.any(r < 1.0):
        raise ValueError("scattering points must lie at r >= 1 solar radius")
    sin_omega = 1.0 / r
    cos_omega = np.sqrt(np.clip(1.0 - sin_omega**2, 0.0, None))
    return sin_omega, cos_omega


def van_de_hulst_coefficients(r):
    """The four geometric coefficients A, B, C, D at radius ``r`` [solar radii].

    These are the disk integral already done: they fold in the finite angular
    size of the Sun and its limb darkening, leaving only closed-form functions
    of ``omega``.

    Returns
    -------
    (A, B, C, D) : tuple of ndarray
        ``A`` and ``C`` carry the uniform-disk part, ``B`` and ``D`` the
        limb-darkened part.
    """
    s, c = _omega_terms(r)

    # ln((1 + sin w) / cos w) — diverges at r = 1 where cos w -> 0, which is
    # inside the photosphere and never sampled by a real line of sight.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.log((1.0 + s) / np.where(c > 0, c, np.nan))

        a = c * s**2
        b = -(1.0 / 8.0) * (
            1.0 - 3.0 * s**2 - (c**2 / s) * (1.0 + 3.0 * s**2) * log_term
        )
        cc = (4.0 / 3.0) - c - (c**3) / 3.0
        d = (1.0 / 8.0) * (
            5.0 + s**2 - (c**2 / s) * (5.0 - s**2) * log_term
        )

    return a, b, cc, d


def scattering_intensities(r, cos_chi, *, u: float = LIMB_DARKENING_U):
    """Tangential and radial scattered intensity per electron, up to a constant.

    ``cos_chi`` is the cosine of the scattering angle between the incoming
    photospheric ray and the line of sight. The returned values carry the full
    geometry; the overall constant (mean solar brightness times the cross
    section) is a scale factor, and this project compares *shape*, so it is
    folded into a single normalisation at the end.

    Returns
    -------
    (i_tan, i_rad) : tuple of ndarray
        Polarised perpendicular to, and within, the scattering plane.
    """
    a, b, c, d = van_de_hulst_coefficients(r)
    sin2_chi = np.clip(1.0 - np.asarray(cos_chi, dtype=np.float64) ** 2, 0.0, 1.0)

    i_tan = (1.0 - u) * c + u * d
    i_rad = i_tan - ((1.0 - u) * a + u * b) * sin2_chi
    return i_tan, i_rad


def total_brightness(r, cos_chi, *, u: float = LIMB_DARKENING_U):
    """Total brightness kernel ``B = I_tan + I_rad``.

    This is what a camera without a polariser records.
    """
    i_tan, i_rad = scattering_intensities(r, cos_chi, u=u)
    return i_tan + i_rad


def polarised_brightness(r, cos_chi, *, u: float = LIMB_DARKENING_U):
    """Polarised brightness kernel ``pB = I_tan - I_rad``.

    Physically the cleaner observable: the F-corona (dust) is nearly
    unpolarised, so ``pB`` isolates the K-corona, which is what this model
    actually computes. Predictive Science publish ``pB`` too, so it is also the
    cleaner comparison channel.
    """
    i_tan, i_rad = scattering_intensities(r, cos_chi, u=u)
    return i_tan - i_rad


def point_source_brightness(r, cos_chi):
    """The textbook point-source limit, ``propto (1 + cos^2 chi) / r^2``.

    Kept as the reference the full kernel must converge to far from the Sun.
    Using *this* in the renderer would be the classic mistake — near the limb,
    where eclipse structure lives, the disk subtends tens of degrees.
    """
    r = np.asarray(r, dtype=np.float64)
    cos_chi = np.asarray(cos_chi, dtype=np.float64)
    return (1.0 + cos_chi**2) / r**2
