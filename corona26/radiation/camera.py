"""Observer geometry: turning a 3-D corona into what a camera would record.

A 3-D density cube is not an image. It has to be projected from where we
actually stand, with the Sun oriented the way it will actually be oriented.

Frame convention used throughout, all in solar radii:

* the Sun is at the origin;
* the observer sits at ``+Z``, looking back down the ``-Z`` direction;
* ``+X`` is image right, which for a solar-north-up view is **solar west**;
* ``+Y`` is image up, which is **solar north**.

The rotation axis is then tilted out of the image plane by ``B0``, the
heliographic latitude of disk centre. Two unit vectors follow:

* ``axis``  = (0, cos B0, sin B0) — solar rotation axis;
* ``e_toward`` = (0, -sin B0, cos B0) — points at the disk-centre meridian.

Their cross product is ``+X``, which is why image right is the west limb, and
why the west limb sits at Carrington longitude ``L0 + 90``.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body
from astropy.time import Time

from corona26.geometry import COLMENAR_VIEJO, TOTALITY, observer_geometry

# Earth-Sun distance in solar radii; the exact value barely matters for the
# projection but it does set how nearly parallel the rays are.
AU_IN_SOLAR_RADII = 215.032


@dataclass(frozen=True)
class Camera:
    """A pinhole view of the corona, in solar radii on the sky."""

    n_pixels: int
    field_of_view: float      # half-width in solar radii
    b0_deg: float
    l0_deg: float
    p_deg: float
    observer_distance: float = AU_IN_SOLAR_RADII
    roll_deg: float = 0.0
    """Angle of solar north from image up, positive clockwise.

    Zero gives the solar-north-up view that is comparable with published
    predictions. Setting it to `solar_north_from_zenith()` gives the
    horizon-referenced view an observer actually sees.
    """

    @property
    def pixel_scale(self) -> float:
        return 2.0 * self.field_of_view / self.n_pixels

    def display_to_solar(self, u_img, v_img):
        """Rotate display-plane coordinates into the solar image frame.

        With solar north at ``roll`` clockwise from image up, the solar basis
        is ``y_hat = cos(roll) up + sin(roll) right`` and ``x_hat`` is 90
        degrees clockwise from it, which inverts to the pair below.
        """
        theta = np.radians(self.roll_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x = u_img * cos_t - v_img * sin_t
        y = u_img * sin_t + v_img * cos_t
        return x, y

    def image_axes(self):
        """Image-plane coordinates in solar radii, centred on the Sun."""
        half = self.field_of_view
        axis = np.linspace(-half + self.pixel_scale / 2,
                           half - self.pixel_scale / 2, self.n_pixels)
        return axis, axis


def unit_vectors(b0_deg: float):
    """Rotation axis and disk-centre direction for a given ``B0``."""
    b0 = np.radians(b0_deg)
    axis = np.array([0.0, np.cos(b0), np.sin(b0)])
    toward = np.array([0.0, -np.sin(b0), np.cos(b0)])
    return axis, toward


def cartesian_to_heliographic(x, y, z, *, b0_deg: float, l0_deg: float):
    """Convert observer-frame Cartesian points to ``(r, sin_lat, lon)``.

    ``lon`` is Carrington longitude in degrees, increasing towards the west
    limb, matching the convention used for the magnetogram everywhere else.
    """
    axis, toward = unit_vectors(b0_deg)
    r = np.sqrt(x * x + y * y + z * z)
    safe = np.where(r > 0, r, 1.0)

    sin_lat = (x * axis[0] + y * axis[1] + z * axis[2]) / safe
    sin_lat = np.clip(sin_lat, -1.0, 1.0)

    along_toward = x * toward[0] + y * toward[1] + z * toward[2]
    along_west = x  # e_west = axis x toward = (1, 0, 0)
    lon = l0_deg + np.degrees(np.arctan2(along_west, along_toward))
    return r, sin_lat, np.mod(lon, 360.0)


def scattering_cosine(x, y, z, observer_distance: float):
    """Cosine of the scattering angle at each point.

    The scattering angle ``chi`` is between the incoming ray (Sun to electron)
    and the outgoing ray (electron to observer). At ``chi = 90 deg`` — the
    Thomson surface — polarisation is maximal.
    """
    r = np.sqrt(x * x + y * y + z * z)
    safe = np.where(r > 0, r, 1.0)

    # Outgoing direction: electron -> observer, who sits at (0, 0, +d).
    ox, oy, oz = -x, -y, observer_distance - z
    onorm = np.sqrt(ox * ox + oy * oy + oz * oz)
    onorm = np.where(onorm > 0, onorm, 1.0)

    # Incoming direction is radial, Sun -> electron.
    return (x * ox + y * oy + z * oz) / (safe * onorm)


def solar_north_from_zenith(
    time: Time = TOTALITY, location: EarthLocation = COLMENAR_VIEJO
) -> u.Quantity:
    """Angle of solar north from the local vertical, positive towards the right.

    This is what turns a solar-north-up render into what a camera on a tripod
    actually records. It is *measured*, not assembled from the position angle
    and the parallactic angle: those two combine with a sign convention that is
    easy to get backwards, and getting it backwards produces a beautiful image
    rotated by the wrong amount with nothing to flag it.

    Method: project the solar north pole and the disk centre into the local
    horizontal frame and take the angle of the offset between them. Facing the
    Sun, azimuth increases to the right, so a positive result tilts solar north
    clockwise from vertical.
    """
    from sunpy.coordinates import frames

    altaz = AltAz(obstime=time, location=location)
    pole = SkyCoord(
        0 * u.deg, 90 * u.deg, 1 * u.R_sun,
        frame=frames.HeliographicStonyhurst, obstime=time, observer="earth",
    ).transform_to(altaz)
    centre = SkyCoord(
        0 * u.deg, 0 * u.deg, 0 * u.R_sun,
        frame=frames.HeliographicStonyhurst, obstime=time, observer="earth",
    ).transform_to(altaz)

    d_alt = (pole.alt - centre.alt).to_value(u.arcsec)
    d_az = ((pole.az - centre.az).to_value(u.arcsec)) * np.cos(
        centre.alt.to_value(u.rad)
    )
    return np.degrees(np.arctan2(d_az, d_alt)) * u.deg


def camera_for_totality(
    n_pixels: int = 768, field_of_view: float = 3.0, *, horizon_referenced: bool = False
) -> Camera:
    """A camera pointed at the Sun from Colmenar Viejo at second contact.

    ``horizon_referenced=False`` puts solar north up, which is what published
    predictions show and what our comparisons use. ``True`` rolls the frame so
    up is the local vertical — what you see standing in the field.
    """
    geom = observer_geometry()
    roll = float(solar_north_from_zenith().to_value(u.deg)) if horizon_referenced else 0.0
    return Camera(
        n_pixels=n_pixels,
        field_of_view=field_of_view,
        b0_deg=float(geom.b0.to_value(u.deg)),
        l0_deg=float(geom.l0.to_value(u.deg)),
        p_deg=float(geom.p_angle.to_value(u.deg)),
        roll_deg=roll,
    )
