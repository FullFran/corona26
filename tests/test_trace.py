"""Tests for magnetic topology.

The two failure modes that do not raise: a seed grid that is not equal-area
(so "open fraction" is not an area fraction), and integration that runs out of
steps (so trapped lines are misread as escaping).
"""

import astropy.units as u
import numpy as np
import pytest

from corona26.magnetic.pfss import dipole_map, solve
from corona26.magnetic.trace import (
    classification_is_converged,
    photospheric_seeds,
    trace_topology,
)


@pytest.fixture(scope="module")
def dipole_output():
    return solve(dipole_map(n_lat=90, n_lon=180), rss=2.5, nr=60).output


class TestSeeding:
    def test_grid_is_equal_area(self, dipole_output):
        """Uniform in sin(latitude), not in latitude.

        If this regresses, polar cells get over-sampled and every open-field
        statistic silently stops being an area fraction.
        """
        seeds = photospheric_seeds(dipole_output.coordinate_frame,
                                   n_lat=90, n_lon=1)
        sin_lat = np.sin(seeds.lat.to_value(u.rad))
        spacing = np.diff(np.sort(sin_lat))
        np.testing.assert_allclose(spacing, spacing[0], rtol=1e-9)

    def test_is_not_uniform_in_latitude(self, dipole_output):
        seeds = photospheric_seeds(dipole_output.coordinate_frame,
                                   n_lat=90, n_lon=1)
        lat = np.sort(seeds.lat.to_value(u.deg))
        spacing = np.diff(lat)
        assert spacing.max() / spacing.min() > 2

    def test_seeds_sit_above_the_boundary(self, dipole_output):
        seeds = photospheric_seeds(dipole_output.coordinate_frame, n_lat=4, n_lon=4)
        assert np.all(seeds.radius > 1.0 * u.R_sun)

    def test_covers_all_longitudes_without_duplicating_the_seam(
        self, dipole_output
    ):
        seeds = photospheric_seeds(dipole_output.coordinate_frame,
                                   n_lat=1, n_lon=8)
        lon = np.sort(seeds.lon.to_value(u.deg))
        assert lon[0] == pytest.approx(0.0)
        assert lon[-1] == pytest.approx(315.0)


class TestDipoleTopology:
    """A dipole has an exactly known topology: open poles, closed equator."""

    @pytest.fixture(scope="class")
    def topology(self, dipole_output):
        return trace_topology(dipole_output, n_lat=40, n_lon=20)

    def test_poles_are_open(self, topology):
        assert topology.is_open[0, :].all()
        assert topology.is_open[-1, :].all()

    def test_equator_is_closed(self, topology):
        mid = topology.n_lat // 2
        assert not topology.is_open[mid - 1 : mid + 1, :].any()

    def test_open_regions_are_the_two_polar_caps(self, topology):
        """Openness must be monotonic in |latitude| for a dipole.

        Reading down a column of longitude there should be exactly two
        transitions: open cap, closed belt, open cap.
        """
        for j in range(topology.n_lon):
            transitions = np.sum(np.diff(topology.is_open[:, j].astype(int)) != 0)
            assert transitions == 2

    def test_the_two_caps_have_opposite_polarity(self, topology):
        north = topology.polarity[-1, :]
        south = topology.polarity[0, :]
        assert np.all(north == -south)
        assert np.all(north != 0)

    def test_is_axisymmetric(self, topology):
        """A dipole has no longitude dependence."""
        for j in range(1, topology.n_lon):
            np.testing.assert_array_equal(
                topology.is_open[:, 0], topology.is_open[:, j]
            )

    def test_open_fraction_is_a_sensible_area_fraction(self, topology):
        assert 0.0 < topology.open_area_fraction < 1.0

    def test_expansion_factor_defined_only_where_open(self, topology):
        assert np.all(np.isfinite(topology.expansion_factor[topology.is_open]))
        assert np.all(np.isnan(topology.expansion_factor[~topology.is_open]))

    def test_summary_is_json_serialisable(self, topology):
        import json

        json.dumps(topology.summary())


class TestSourceSurfaceDependence:
    def test_lower_source_surface_opens_more_flux(self, dipole_output):
        """More of the surface is open when the wind reaches deeper."""
        m = dipole_map(n_lat=60, n_lon=120)
        low = trace_topology(solve(m, rss=1.5, nr=40).output, n_lat=40, n_lon=10)
        high = trace_topology(solve(m, rss=3.0, nr=40).output, n_lat=40, n_lon=10)
        assert low.open_area_fraction > high.open_area_fraction


class TestConvergence:
    def test_classification_does_not_depend_on_the_step_budget(
        self, dipole_output
    ):
        converged, changed = classification_is_converged(
            dipole_output, n_lat=30, n_lon=12, max_steps=2000
        )
        assert converged, f"{changed} seeds changed class when steps doubled"
