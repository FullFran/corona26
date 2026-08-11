"""Observer geometry: where we are and how the Sun is oriented from there.

Nothing here is hardcoded from a table. Every angle is computed from
ephemerides, so changing the site or the time changes the whole pipeline
consistently.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, get_body
from astropy.time import Time
from sunpy.coordinates import sun

# Totality at Colmenar Viejo: 2026-08-12 20:31 CEST == 18:31 UTC.
TOTALITY = Time("2026-08-12T18:31:00", scale="utc")

COLMENAR_VIEJO = EarthLocation(
    lat=40.6591 * u.deg,
    lon=-3.7676 * u.deg,
    height=1004 * u.m,
)


@dataclass(frozen=True)
class ObserverGeometry:
    """Solar orientation and sky position for one site at one instant."""

    time: Time
    p_angle: u.Quantity      # position angle of solar north, east of celestial north
    b0: u.Quantity           # heliographic latitude of disk centre
    l0: u.Quantity           # Carrington longitude of disk centre
    angular_radius: u.Quantity
    altitude: u.Quantity
    azimuth: u.Quantity

    @property
    def east_limb_longitude(self) -> u.Quantity:
        """Carrington longitude of the east limb.

        The Carrington longitude of the central meridian decreases with time,
        so material reaches disk centre from lower longitudes. The east limb —
        the hemisphere edge most recently emerged from the far side, and
        therefore the least constrained by observation — is at ``L0 - 90``.
        """
        return (self.l0 - 90 * u.deg) % (360 * u.deg)

    @property
    def west_limb_longitude(self) -> u.Quantity:
        return (self.l0 + 90 * u.deg) % (360 * u.deg)

    def summary(self) -> dict[str, float]:
        return {
            "time_utc": self.time.isot,
            "P_deg": float(self.p_angle.to_value(u.deg)),
            "B0_deg": float(self.b0.to_value(u.deg)),
            "L0_deg": float(self.l0.to_value(u.deg)),
            "angular_radius_arcmin": float(self.angular_radius.to_value(u.arcmin)),
            "altitude_deg": float(self.altitude.to_value(u.deg)),
            "azimuth_deg": float(self.azimuth.to_value(u.deg)),
        }


def observer_geometry(
    time: Time = TOTALITY,
    location: EarthLocation = COLMENAR_VIEJO,
) -> ObserverGeometry:
    """Compute solar orientation and horizon position for a site and time."""
    altaz = get_body("sun", time, location=location).transform_to(
        AltAz(obstime=time, location=location)
    )
    return ObserverGeometry(
        time=time,
        p_angle=sun.P(time).to(u.deg),
        b0=sun.B0(time).to(u.deg),
        l0=sun.L0(time).to(u.deg),
        angular_radius=sun.angular_radius(time).to(u.arcmin),
        altitude=altaz.alt.to(u.deg),
        azimuth=altaz.az.to(u.deg),
    )
