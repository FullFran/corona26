"""Tests for the PFSS reconstruction.

The solver is a library, so these are not testing someone else's maths. They
are testing that *we drive it correctly* — sign conventions, projection,
boundary conditions — because every one of those fails silently and returns a
plausible corona.
"""

import numpy as np
import pytest

from corona26.magnetic.pfss import (
    RSS_ENSEMBLE,
    boundary_residual,
    cache_key,
    dipole_br,
    dipole_map,
    open_flux,
    solve,
)


class TestAnalyticDipole:
    """The closed form the numerical solution must reproduce."""

    def test_reduces_to_the_boundary_condition_at_the_photosphere(self):
        theta = np.linspace(0.01, np.pi - 0.01, 50)
        np.testing.assert_allclose(dipole_br(theta, 1.0, rss=2.5), np.cos(theta))

    def test_field_is_purely_radial_and_falls_off_at_the_source_surface(self):
        # Br at Rss must be positive-definite in sign with cos(theta) and
        # weaker than at the surface.
        theta = np.array([0.1])
        assert dipole_br(theta, 2.5, rss=2.5) < dipole_br(theta, 1.0, rss=2.5)

    def test_vanishes_at_the_equator(self):
        assert dipole_br(np.array([np.pi / 2]), 1.5, rss=2.5) == pytest.approx(0.0)

    def test_antisymmetric_about_the_equator(self):
        theta = np.linspace(0.01, np.pi / 2 - 0.01, 20)
        north = dipole_br(theta, 1.7, rss=2.5)
        south = dipole_br(np.pi - theta, 1.7, rss=2.5)
        np.testing.assert_allclose(north, -south, atol=1e-12)

    def test_larger_source_surface_gives_a_steeper_falloff(self):
        """As Rss grows the field approaches the unconfined dipole r^-3."""
        theta = np.array([0.0])
        ratio_small = dipole_br(theta, 1.3, rss=1.3) / dipole_br(theta, 1.0, rss=1.3)
        ratio_large = dipole_br(theta, 1.3, rss=3.0) / dipole_br(theta, 1.0, rss=3.0)
        assert ratio_large < ratio_small

    def test_rejects_radii_outside_the_domain(self):
        with pytest.raises(ValueError):
            dipole_br(np.array([0.5]), 0.5, rss=2.5)
        with pytest.raises(ValueError):
            dipole_br(np.array([1.0]), 5.0, rss=2.5)


class TestDipoleMap:
    def test_is_cea_and_the_right_shape(self):
        m = dipole_map(n_lat=90, n_lon=180)
        assert m.data.shape == (90, 180)
        assert "CEA" in m.meta["ctype2"].upper()

    def test_is_antisymmetric_so_carries_no_net_flux(self):
        m = dipole_map()
        assert np.sum(m.data) == pytest.approx(0.0, abs=1e-9)


class TestSolverAgainstAnalyticDipole:
    """The end-to-end check: drive the real solver with a dipole."""

    @pytest.fixture(scope="class")
    def solution(self):
        return solve(dipole_map(n_lat=90, n_lon=180), rss=2.5, nr=60)

    def test_reproduces_the_input_boundary(self, solution):
        """The solver ingests the boundary essentially exactly.

        A loose threshold here would hide a staggering error: reading Br from
        the corner-centred `bg` array instead of the cell-centred `bc` gives a
        ~27% residual that is easy to misread as harmonic truncation.
        """
        residual = boundary_residual(solution, dipole_map(n_lat=90, n_lon=180))
        assert residual < 0.001

    def test_source_surface_field_matches_the_closed_form(self, solution):
        """Sign, normalisation and upper boundary condition, all at once."""
        ss = np.asarray(solution.source_surface_br.data, dtype=np.float64)
        n_lat = ss.shape[0]
        # CEA rows are uniform in sin(latitude).
        sin_lat = 0.5 * (
            np.linspace(-1, 1, n_lat + 1)[:-1] + np.linspace(-1, 1, n_lat + 1)[1:]
        )
        theta = np.arccos(sin_lat)
        expected = dipole_br(theta, 2.5, rss=2.5)
        measured = np.nanmean(ss, axis=1)  # axisymmetric, so average over longitude
        np.testing.assert_allclose(measured, expected, rtol=0.05, atol=1e-3)

    def test_polarity_is_not_inverted(self, solution):
        """The one error that produces a beautiful, completely wrong corona."""
        ss = np.asarray(solution.source_surface_br.data, dtype=np.float64)
        northern = np.nanmean(ss[ss.shape[0] // 2 :, :])
        southern = np.nanmean(ss[: ss.shape[0] // 2, :])
        assert northern > 0 > southern

    def test_open_flux_is_positive_and_finite(self, solution):
        assert 0 < open_flux(solution) < np.inf


class TestSourceSurfaceDependence:
    def test_open_flux_falls_as_the_source_surface_rises(self):
        """Physics sanity: a higher source surface confines more flux.

        Raising Rss lets more field close below it, so less flux is open. If
        this ever inverts, the ensemble is exploring the parameter backwards.
        """
        m = dipole_map(n_lat=60, n_lon=120)
        fluxes = [open_flux(solve(m, rss=rss, nr=40)) for rss in (1.5, 3.0)]
        assert fluxes[0] > fluxes[1]


class TestGuards:
    def test_rejects_a_plate_carree_map(self):
        m = dipole_map()
        m.meta["ctype1"] = "CRLN-CAR"
        m.meta["ctype2"] = "CRLT-CAR"
        with pytest.raises(ValueError, match="CEA"):
            solve(m)

    def test_ensemble_covers_both_sides_of_the_default(self):
        assert min(RSS_ENSEMBLE) < 2.5 < max(RSS_ENSEMBLE)


class TestCacheKey:
    def test_is_stable(self):
        assert cache_key("a.fts", 0, 2.5, 100) == cache_key("a.fts", 0, 2.5, 100)

    def test_distinguishes_every_parameter(self):
        base = cache_key("a.fts", 0, 2.5, 100)
        assert base != cache_key("b.fts", 0, 2.5, 100)
        assert base != cache_key("a.fts", 1, 2.5, 100)
        assert base != cache_key("a.fts", 0, 3.0, 100)
        assert base != cache_key("a.fts", 0, 2.5, 50)
