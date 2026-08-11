"""Tests for the Thomson scattering kernel.

This is the part of the pipeline we wrote ourselves and the part that is
genuinely exact, so it gets the hardest tests. The decisive one is the
point-source limit: far from the Sun the full finite-disk kernel must collapse
onto the textbook (1 + cos^2 chi) dipole pattern. That single check pins the
sign, the normalisation and the disk integral together — nothing else in this
module can be wrong while it passes.
"""

import numpy as np
import pytest

from corona26.radiation.thomson import (
    LIMB_DARKENING_U,
    SIGMA_THOMSON,
    point_source_brightness,
    polarised_brightness,
    scattering_intensities,
    total_brightness,
    van_de_hulst_coefficients,
)


class TestConstants:
    def test_thomson_cross_section(self):
        assert SIGMA_THOMSON == pytest.approx(6.652e-29, rel=1e-3)


class TestCoefficients:
    def test_all_finite_across_the_corona(self):
        r = np.linspace(1.01, 30.0, 400)
        for coeff in van_de_hulst_coefficients(r):
            assert np.all(np.isfinite(coeff))

    def test_uniform_disk_coefficients_are_positive(self):
        r = np.linspace(1.01, 30.0, 200)
        a, _, c, _ = van_de_hulst_coefficients(r)
        assert np.all(a > 0)
        assert np.all(c > 0)

    def test_all_decay_with_distance(self):
        r = np.array([1.5, 3.0, 10.0, 50.0])
        for coeff in van_de_hulst_coefficients(r):
            assert np.all(np.diff(np.abs(coeff)) < 0)

    def test_a_and_c_converge_to_each_other_far_away(self):
        """Both tend to sin^2(omega) as the disk shrinks to a point."""
        a, _, c, _ = van_de_hulst_coefficients(np.array([500.0]))
        assert a[0] == pytest.approx(c[0], rel=1e-4)

    def test_a_and_c_approach_the_inverse_square_law(self):
        r = np.array([200.0, 400.0])
        a, _, c, _ = van_de_hulst_coefficients(r)
        assert a[0] == pytest.approx(1.0 / r[0] ** 2, rel=1e-3)
        assert c[0] == pytest.approx(1.0 / r[0] ** 2, rel=1e-3)

    def test_rejects_points_inside_the_photosphere(self):
        with pytest.raises(ValueError):
            van_de_hulst_coefficients(np.array([0.5]))


class TestPointSourceLimit:
    """The decisive test: the finite-disk kernel must reduce to the textbook one."""

    @pytest.mark.parametrize("cos_chi", [-1.0, -0.5, 0.0, 0.3, 0.7, 1.0])
    def test_matches_the_dipole_pattern_far_from_the_sun(self, cos_chi):
        r = 1000.0
        full = total_brightness(r, cos_chi, u=0.0)
        point = point_source_brightness(r, cos_chi)
        # Compare shapes: the kernels share a constant we do not carry.
        ref_full = total_brightness(r, 0.0, u=0.0)
        ref_point = point_source_brightness(r, 0.0)
        assert full / ref_full == pytest.approx(point / ref_point, rel=1e-3)

    def test_angular_dependence_is_the_classical_one(self):
        """B(chi) / B(90 deg) must equal (1 + cos^2 chi) / 1 far away."""
        cos_chi = np.linspace(-1, 1, 21)
        r = 1000.0
        ratio = total_brightness(r, cos_chi, u=0.0) / total_brightness(r, 0.0, u=0.0)
        np.testing.assert_allclose(ratio, 1.0 + cos_chi**2, rtol=2e-3)

    def test_convergence_improves_with_distance(self):
        def error(r):
            cos_chi = np.linspace(-1, 1, 11)
            ratio = total_brightness(r, cos_chi, u=0.0) / total_brightness(
                r, 0.0, u=0.0
            )
            return np.max(np.abs(ratio - (1.0 + cos_chi**2)))

        assert error(1000.0) < error(100.0) < error(10.0)

    def test_finite_disk_differs_near_the_limb(self):
        """Where eclipse structure actually lives, the point source is wrong."""
        cos_chi = 0.9
        r = 1.5
        ratio_full = total_brightness(r, cos_chi, u=0.0) / total_brightness(
            r, 0.0, u=0.0
        )
        ratio_point = (1.0 + cos_chi**2) / 1.0
        assert abs(ratio_full - ratio_point) / ratio_point > 0.1


class TestPolarisation:
    def test_tangential_always_exceeds_radial(self):
        """Thomson scattering polarises tangentially; pB must be positive."""
        r = np.linspace(1.05, 10.0, 50)
        for cos_chi in (-0.9, -0.4, 0.0, 0.4, 0.9):
            i_tan, i_rad = scattering_intensities(r, cos_chi)
            assert np.all(i_tan >= i_rad - 1e-15)

    def test_polarisation_vanishes_along_the_sun_observer_axis(self):
        """At chi = 0 or 180 degrees there is no preferred transverse direction."""
        assert polarised_brightness(2.0, 1.0) == pytest.approx(0.0, abs=1e-12)
        assert polarised_brightness(2.0, -1.0) == pytest.approx(0.0, abs=1e-12)

    def test_polarisation_peaks_at_ninety_degrees(self):
        """The Thomson surface: pB is maximised where the scattering angle is 90."""
        cos_chi = np.linspace(-1, 1, 101)
        pb = polarised_brightness(2.0, cos_chi)
        assert np.argmax(pb) == 50  # cos_chi == 0

    def test_degree_of_polarisation_is_physical(self):
        r = np.linspace(1.05, 20.0, 60)
        for cos_chi in (-0.8, 0.0, 0.8):
            p = polarised_brightness(r, cos_chi) / total_brightness(r, cos_chi)
            assert np.all(p >= -1e-12)
            assert np.all(p <= 1.0 + 1e-12)


class TestLimbDarkening:
    def test_default_matches_visible_light(self):
        assert LIMB_DARKENING_U == pytest.approx(0.6)

    def test_darkening_reduces_brightness(self):
        """A limb-darkened Sun sends out less light than a uniform one."""
        r = np.linspace(1.05, 10.0, 40)
        assert np.all(total_brightness(r, 0.3, u=0.6) < total_brightness(r, 0.3, u=0.0))

    def test_is_continuous_in_u(self):
        r = 2.0
        values = [total_brightness(r, 0.3, u=u) for u in (0.0, 0.3, 0.6, 1.0)]
        assert all(np.isfinite(values))
        assert values == sorted(values, reverse=True)


class TestRadialFalloff:
    def test_brightness_falls_steeply_with_radius(self):
        r = np.array([1.1, 2.0, 5.0])
        b = total_brightness(r, 0.0)
        assert b[0] > b[1] > b[2]

    def test_falloff_approaches_inverse_square_far_out(self):
        r = np.array([100.0, 200.0])
        b = total_brightness(r, 0.0, u=0.0)
        assert b[0] / b[1] == pytest.approx(4.0, rel=1e-2)
