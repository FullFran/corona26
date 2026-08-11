"""Tests for the camera geometry, the density proxy and the renderer.

Orientation errors are the dangerous ones here: they produce a beautiful image
that is rotated, mirrored, or looking at the wrong side of the Sun, and nothing
about it looks wrong. So the geometry is pinned against points whose answer is
known by construction — disk centre, the two limbs, the poles.
"""

import numpy as np
import pytest

from corona26.plasma.radial_density import baumbach_allen, scale_height_slope
from corona26.plasma.topology_density import DensityCube, sample_density
from corona26.radiation.camera import (
    Camera,
    cartesian_to_heliographic,
    scattering_cosine,
)
from corona26.radiation.render import radial_filter, render

L0 = 224.53
B0 = 6.51


def _camera(n=48, fov=3.0):
    return Camera(n_pixels=n, field_of_view=fov, b0_deg=B0, l0_deg=L0, p_deg=15.19)


def _uniform_cube(r_max=3.0, n_r=20, n_lat=24, n_lon=48):
    """A spherically symmetric density: no topology, pure radial profile."""
    r = np.geomspace(1.02, r_max, n_r)
    edges = np.linspace(-1, 1, n_lat + 1)
    sin_lat = 0.5 * (edges[:-1] + edges[1:])
    lon = np.linspace(0, 360, n_lon, endpoint=False)
    ne = np.broadcast_to(
        baumbach_allen(r)[:, None, None], (n_r, n_lat, n_lon)
    ).copy()
    return DensityCube(
        ne=ne, closedness=np.ones_like(ne), r=r, sin_lat=sin_lat, lon=lon, rss=2.5
    )


class TestRadialDensity:
    def test_falls_monotonically(self):
        r = np.linspace(1.01, 10.0, 200)
        assert np.all(np.diff(baumbach_allen(r)) < 0)

    def test_dynamic_range_across_the_rendered_corona(self):
        """Density falls ~170x between 1.1 and 3 solar radii.

        Measured, not assumed. The *brightness* range is steeper still (~3.3
        orders) because the line-of-sight path and the scattering kernel fall
        off too — which is why every eclipse image needs a radial filter.
        """
        ratio = baumbach_allen(1.1) / baumbach_allen(3.0)
        assert 150 < ratio < 200

    def test_slope_is_steep_near_the_surface(self):
        assert scale_height_slope(1.05) < -8

    def test_slope_flattens_towards_the_wind(self):
        """Far out, mass conservation in a radial flow gives ne ~ r^-2."""
        assert -3 < scale_height_slope(50.0) < -1.4

    def test_rejects_points_below_the_photosphere(self):
        with pytest.raises(ValueError):
            baumbach_allen(0.9)


class TestCameraGeometry:
    def test_disk_centre_maps_to_l0(self):
        """A point straight in front of the Sun sits at the disk-centre meridian."""
        _, sin_lat, lon = cartesian_to_heliographic(
            0.0, -np.sin(np.radians(B0)) * 2, np.cos(np.radians(B0)) * 2,
            b0_deg=B0, l0_deg=L0,
        )
        assert lon == pytest.approx(L0, abs=1e-6)
        assert sin_lat == pytest.approx(0.0, abs=1e-9)

    def test_image_right_is_the_west_limb(self):
        """Solar west is image right, and it sits at L0 + 90."""
        _, _, lon = cartesian_to_heliographic(2.0, 0.0, 0.0, b0_deg=B0, l0_deg=L0)
        assert lon == pytest.approx((L0 + 90) % 360, abs=1e-6)

    def test_image_left_is_the_east_limb(self):
        """The badly constrained one, at L0 - 90."""
        _, _, lon = cartesian_to_heliographic(-2.0, 0.0, 0.0, b0_deg=B0, l0_deg=L0)
        assert lon == pytest.approx((L0 - 90) % 360, abs=1e-6)

    def test_rotation_axis_maps_to_the_pole(self):
        b0 = np.radians(B0)
        _, sin_lat, _ = cartesian_to_heliographic(
            0.0, 2 * np.cos(b0), 2 * np.sin(b0), b0_deg=B0, l0_deg=L0
        )
        assert sin_lat == pytest.approx(1.0, abs=1e-9)

    def test_radius_is_recovered(self):
        r, _, _ = cartesian_to_heliographic(1.0, 2.0, 2.0, b0_deg=B0, l0_deg=L0)
        assert r == pytest.approx(3.0)

    def test_b0_tilts_the_pole_towards_the_observer(self):
        """With B0 > 0 the north pole leans towards us, so image up is not the pole."""
        _, sin_lat, _ = cartesian_to_heliographic(0.0, 2.0, 0.0, b0_deg=B0, l0_deg=L0)
        assert sin_lat < 1.0
        assert sin_lat == pytest.approx(np.cos(np.radians(B0)), abs=1e-9)


class TestScatteringGeometry:
    def test_plane_of_sky_scatters_at_ninety_degrees(self):
        """On the plane of the sky the scattering angle is 90 deg, so cos = 0."""
        cos_chi = scattering_cosine(2.0, 0.0, 0.0, 215.032)
        assert cos_chi == pytest.approx(0.0, abs=1e-2)

    def test_in_front_of_the_sun_is_forward_scattering(self):
        assert scattering_cosine(0.0, 0.0, 2.0, 215.032) > 0.99

    def test_behind_the_sun_is_back_scattering(self):
        assert scattering_cosine(0.0, 0.0, -2.0, 215.032) < -0.99

    def test_is_bounded(self):
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(3, 500)) * 2
        c = scattering_cosine(pts[0], pts[1], pts[2], 215.032)
        assert np.all(np.abs(c) <= 1.0 + 1e-12)


class TestDensitySampling:
    def test_returns_zero_outside_the_shell(self):
        cube = _uniform_cube()
        assert sample_density(cube, np.array([0.5]), np.array([0.0]),
                              np.array([0.0]))[0] == 0.0
        assert sample_density(cube, np.array([10.0]), np.array([0.0]),
                              np.array([0.0]))[0] == 0.0

    def test_reproduces_the_radial_profile(self):
        cube = _uniform_cube()
        r = np.array([1.5, 2.0, 2.5])
        got = sample_density(cube, r, np.zeros_like(r), np.zeros_like(r))
        np.testing.assert_allclose(got, baumbach_allen(r), rtol=0.05)

    def test_longitude_wraps(self):
        cube = _uniform_cube()
        r = np.array([2.0])
        a = sample_density(cube, r, np.array([0.0]), np.array([359.9]))
        b = sample_density(cube, r, np.array([0.0]), np.array([-0.1]))
        assert a[0] == pytest.approx(b[0])


class TestRenderer:
    def test_occults_the_lunar_disk(self):
        """Pixels the Moon covers must be exactly zero, not merely dim."""
        out = render(_uniform_cube(), _camera(), n_samples=96)
        x, y = out.camera.image_axes()
        xx, yy = np.meshgrid(x, y)
        inside = np.hypot(xx, yy) < 1.0
        assert np.all(out.total[inside] == 0.0)
        assert np.any(out.total[~inside] > 0.0)

    def test_spherical_density_renders_a_circular_corona(self):
        """No topology in, no angular structure out.

        Tested as invariance under a 90 degree rotation rather than as scatter
        within an annulus: a narrow annulus still spans a real radial gradient,
        so its scatter measures the falloff, not asymmetry.
        """
        out = render(_uniform_cube(), _camera(n=64), n_samples=256)
        rotated = np.rot90(out.total)
        mask = (out.total > 0) & (rotated > 0)
        relative = np.abs(out.total[mask] - rotated[mask]) / out.total[mask]
        assert relative.max() < 1e-9

    def test_brightness_falls_with_height(self):
        out = render(_uniform_cube(), _camera(n=64), n_samples=256)
        x, y = out.camera.image_axes()
        xx, yy = np.meshgrid(x, y)
        radius = np.hypot(xx, yy)

        def ring_mean(lo, hi):
            sel = (radius > lo) & (radius < hi) & (out.total > 0)
            return out.total[sel].mean()

        assert ring_mean(1.05, 1.2) > ring_mean(1.5, 1.7) > ring_mean(2.3, 2.6)

    def test_polarised_is_positive_and_below_total(self):
        out = render(_uniform_cube(), _camera(n=48), n_samples=192)
        visible = out.total > 0
        assert np.all(out.polarised[visible] > 0)
        assert np.all(out.polarised[visible] < out.total[visible])

    def test_degree_of_polarisation_is_physical(self):
        out = render(_uniform_cube(), _camera(n=48), n_samples=192)
        visible = out.total > 0
        p = out.polarised[visible] / out.total[visible]
        assert np.all((p > 0.0) & (p < 1.0))

    def test_quadrature_is_converged_on_the_visible_corona(self):
        cube, cam = _uniform_cube(), _camera(n=48)
        coarse = render(cube, cam, n_samples=384).total
        fine = render(cube, cam, n_samples=1536).total
        mask = fine > 0
        error = np.abs(coarse[mask] - fine[mask]) / fine[mask]
        assert np.median(error) < 1e-4
        assert error.max() < 0.03

    def test_tiling_does_not_change_the_result(self):
        cube, cam = _uniform_cube(), _camera(n=64)
        whole = render(cube, cam, n_samples=192, tile=64).total
        tiled = render(cube, cam, n_samples=192, tile=16).total
        np.testing.assert_allclose(whole, tiled, rtol=1e-12)


class TestRadialFilter:
    def test_flattens_the_radial_falloff(self):
        cube, cam = _uniform_cube(), _camera(n=64)
        out = render(cube, cam, n_samples=256)
        filtered = radial_filter(out.total, cam)

        x, y = cam.image_axes()
        xx, yy = np.meshgrid(x, y)
        radius = np.hypot(xx, yy)

        def ring_mean(img, lo, hi):
            sel = (radius > lo) & (radius < hi) & (img > 0)
            return img[sel].mean()

        raw_ratio = ring_mean(out.total, 1.1, 1.3) / ring_mean(out.total, 2.2, 2.4)
        filt_ratio = ring_mean(filtered, 1.1, 1.3) / ring_mean(filtered, 2.2, 2.4)
        assert raw_ratio > 50
        assert 0.5 < filt_ratio < 2.0

    def test_keeps_the_disk_occulted(self):
        cube, cam = _uniform_cube(), _camera(n=48)
        filtered = radial_filter(render(cube, cam, n_samples=96).total, cam)
        x, y = cam.image_axes()
        xx, yy = np.meshgrid(x, y)
        assert np.all(filtered[np.hypot(xx, yy) < 1.0] == 0.0)
