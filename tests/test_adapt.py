"""Tests for the ADAPT boundary condition.

These target the two things that fail silently: area weighting on an
equal-angle grid, and the missing-value sentinel.
"""

import numpy as np
import pytest

from corona26.data.adapt import (
    MISSING_VALUE,
    _clean,
    flux_balance,
    latitude_centres,
)

SHAPE = (180, 360)


class TestLatitudeCentres:
    def test_spans_the_poles_without_touching_them(self):
        lat = latitude_centres(180)
        assert lat.shape == (180,)
        assert -90 < lat.min() < -89
        assert 89 < lat.max() < 90

    def test_symmetric_about_the_equator(self):
        lat = latitude_centres(180)
        np.testing.assert_allclose(lat, -lat[::-1], atol=1e-12)

    def test_no_row_sits_exactly_on_a_pole(self):
        # cos(lat) would vanish and silently zero-weight a whole row.
        assert np.all(np.abs(latitude_centres(180)) < 90)


class TestFluxBalance:
    def test_uniform_field_is_pure_monopole(self):
        br = np.ones(SHAPE)
        result = flux_balance(br)
        assert result["monopole_ratio"] == pytest.approx(1.0)

    def test_antisymmetric_dipole_has_no_net_flux(self):
        lat = np.deg2rad(latitude_centres(SHAPE[0]))
        br = np.sin(lat)[:, None] * np.ones((1, SHAPE[1]))
        result = flux_balance(br)
        assert result["monopole_ratio"] == pytest.approx(0.0, abs=1e-12)
        assert result["unsigned_flux"] > 0

    def test_area_weighting_is_applied(self):
        """Equal-angle cells do not have equal area.

        The same field amplitude placed in a polar row must contribute far
        less flux than in an equatorial row. Without the cos(lat) weight this
        test fails and every flux diagnostic downstream is wrong.
        """
        equatorial = np.zeros(SHAPE)
        equatorial[90, :] = 1.0
        polar = np.zeros(SHAPE)
        polar[1, :] = 1.0

        assert (
            flux_balance(polar)["unsigned_flux"]
            < 0.05 * flux_balance(equatorial)["unsigned_flux"]
        )

    def test_nans_are_excluded_not_propagated(self):
        br = np.ones(SHAPE)
        br[0, 0] = np.nan
        result = flux_balance(br)
        assert np.isfinite(result["signed_flux"])
        assert result["monopole_ratio"] == pytest.approx(1.0)

    def test_rejects_a_full_cube(self):
        with pytest.raises(ValueError, match="2-D"):
            flux_balance(np.ones((12, *SHAPE)))


class TestClean:
    def test_sentinel_becomes_nan(self):
        raw = np.array([[1.0, MISSING_VALUE], [3.0, 4.0]])
        cleaned = _clean(raw)
        assert np.isnan(cleaned[0, 1])
        assert cleaned[0, 0] == 1.0

    def test_does_not_mutate_the_input(self):
        raw = np.array([[MISSING_VALUE]])
        _clean(raw)
        assert raw[0, 0] == MISSING_VALUE

    def test_promotes_to_float64(self):
        # ADAPT ships big-endian float32; downstream maths wants native f8.
        raw = np.ones((2, 2), dtype=">f4")
        assert _clean(raw).dtype == np.float64
