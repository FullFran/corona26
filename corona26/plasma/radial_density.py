"""Radial electron density: an empirical profile, honestly labelled.

PFSS gives no plasma. To render anything we have to say where the electrons
are, and nothing in the magnetic model tells us. So this module is a *proxy*:
a published empirical fit, with its parameters exposed rather than buried.

The canonical choice is Baumbach (1937) / Allen (1947), which is the right one
here for a pleasing reason — it was derived from *eclipse* white-light
photometry in the first place. Using it to predict an eclipse is closing a
loop that is almost a century old.
"""

from __future__ import annotations

import numpy as np

# Baumbach-Allen, electrons per cm^3, r in solar radii. The three terms are
# not arbitrary: the steep ones describe gravitationally stratified plasma
# near the surface, the shallow one the asymptotic radial solar wind.
BAUMBACH_ALLEN = ((2.99, 16.0), (1.55, 6.0), (0.036, 1.5))
BAUMBACH_ALLEN_SCALE = 1.0e8


def baumbach_allen(r):
    """Electron density [cm^-3] from the classic eclipse-derived fit.

    ``ne(r) = 1e8 (2.99 r^-16 + 1.55 r^-6 + 0.036 r^-1.5)``
    """
    r = np.asarray(r, dtype=np.float64)
    if np.any(r < 1.0):
        raise ValueError("density is only defined above the photosphere")
    total = np.zeros_like(r)
    for coefficient, exponent in BAUMBACH_ALLEN:
        total = total + coefficient * r ** (-exponent)
    return BAUMBACH_ALLEN_SCALE * total


def power_law_sum(r, *, a2: float = 0.036, a4: float = 0.5, a6: float = 1.55):
    """The alternative ``A r^-2 + B r^-4 + C r^-6`` form, in units of 1e8 cm^-3.

    Kept because it is the form quoted most often in modelling papers, and
    because swapping profiles is a cheap sensitivity test on a part of the
    pipeline we know is weak.
    """
    r = np.asarray(r, dtype=np.float64)
    if np.any(r < 1.0):
        raise ValueError("density is only defined above the photosphere")
    return BAUMBACH_ALLEN_SCALE * (a2 * r**-2.0 + a4 * r**-4.0 + a6 * r**-6.0)


def scale_height_slope(r, profile=baumbach_allen, *, h: float = 1e-3):
    """Local logarithmic slope ``d ln(ne) / d ln(r)``.

    A quick diagnostic: the corona should fall off steeply near the surface
    (slope around -8 or steeper) and flatten towards the wind's -2 far out.
    """
    r = np.asarray(r, dtype=np.float64)
    upper = np.log(profile(r * (1 + h)))
    lower = np.log(profile(r * (1 - h)))
    return (upper - lower) / (2.0 * h)
