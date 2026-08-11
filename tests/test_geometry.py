"""Tests for observer geometry and the limb-longitude convention."""

import astropy.units as u
import pytest
from astropy.time import Time

from corona26.geometry import TOTALITY, observer_geometry
from corona26.plotting.magnetogram import limb_longitudes


@pytest.fixture(scope="module")
def geom():
    return observer_geometry()


class TestTotalityGeometry:
    def test_sun_is_low_in_the_west_northwest(self, geom):
        # If this ever fails, either the time or the site is wrong — and the
        # whole render orientation is wrong with it.
        assert 5 * u.deg < geom.altitude < 10 * u.deg
        assert 275 * u.deg < geom.azimuth < 290 * u.deg

    def test_b0_is_northward_in_august(self, geom):
        # Earth passes the ascending node in early June; B0 peaks in September.
        assert 5 * u.deg < geom.b0 < 8 * u.deg

    def test_angular_radius_near_aphelion(self, geom):
        assert 15.7 * u.arcmin < geom.angular_radius < 15.9 * u.arcmin

    def test_summary_is_json_serialisable(self, geom):
        import json

        json.dumps(geom.summary())


class TestLimbConvention:
    def test_east_limb_is_ninety_degrees_below_disk_centre(self, geom):
        expected = (geom.l0 - 90 * u.deg) % (360 * u.deg)
        assert geom.east_limb_longitude == expected

    def test_limbs_are_opposite(self, geom):
        separation = (geom.west_limb_longitude - geom.east_limb_longitude) % (
            360 * u.deg
        )
        assert separation.to_value(u.deg) == pytest.approx(180.0)

    def test_carrington_longitude_decreases_with_time(self):
        """The convention the east/west labelling depends on.

        If this ever inverts, `east limb` and `west limb` swap and the
        uncertainty argument in the README points at the wrong side of the
        corona.
        """
        later = observer_geometry(TOTALITY + 1 * u.day)
        earlier = observer_geometry(TOTALITY)
        assert later.l0 < earlier.l0

    def test_helper_matches_the_dataclass(self, geom):
        helper = limb_longitudes(geom.l0.to_value(u.deg))
        assert helper["east limb"] == pytest.approx(
            geom.east_limb_longitude.to_value(u.deg)
        )
        assert helper["west limb"] == pytest.approx(
            geom.west_limb_longitude.to_value(u.deg)
        )

    def test_wraps_across_zero(self):
        limbs = limb_longitudes(45.0)
        assert limbs["east limb"] == pytest.approx(315.0)
        assert limbs["west limb"] == pytest.approx(135.0)


def test_map_time_precedes_totality():
    """A prediction must be built from data older than the event."""
    assert Time("2026-08-11T04:00:00") < TOTALITY
