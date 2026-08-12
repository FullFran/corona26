"""Focused tests for the locked post-eclipse validation contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from corona26.validation.alignment import (
    Alignment,
    Crop,
    align_north_up,
    circular_distance_deg,
    position_angle_deg,
)
from corona26.validation.metrics import match_peaks, score_profile_pair
from corona26.validation.profiles import (
    AngularProfile,
    detect_streamer_peaks,
    sample_angular_profile,
)
from corona26.validation.provenance import (
    load_canonical_protocol_lock,
    load_frozen_prediction,
    load_json,
    sha256_file,
    validate_observation_manifest,
    validate_alignment_for_image,
    validate_official_score,
    validate_prediction_manifest,
    validate_protocol_lock,
)

ROOT = Path(__file__).resolve().parents[1]


def _profile(peaks: tuple[float, ...], *, gain=1.0, offset=0.0) -> AngularProfile:
    pa = np.arange(360.0)
    values = np.zeros_like(pa)
    for peak in peaks:
        distance = circular_distance_deg(pa, peak)
        values += np.exp(-0.5 * (distance / 4.0) ** 2)
    values = gain * values + offset
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    normalized = (values - median) / mad
    return AngularProfile(2.0, pa, normalized, 1.0, median, mad, "ok")


class TestPositionAngle:
    @pytest.mark.parametrize(
        ("x", "y", "expected"),
        [(10, 0, 0), (0, 10, 90), (10, 20, 180), (20, 10, 270)],
    )
    def test_cardinal_directions(self, x, y, expected):
        assert position_angle_deg(x, y, center_x=10, center_y=10) == expected

    def test_wrap_distance(self):
        assert circular_distance_deg(359, 1) == pytest.approx(2)
        assert circular_distance_deg(10, 190) == pytest.approx(180)


class TestCropAndAlignment:
    def test_explicit_crop_is_half_open_and_copied(self):
        image = np.arange(6 * 8).reshape(6, 8)
        cropped = Crop(2, 1, 6, 5).apply(image)
        np.testing.assert_array_equal(cropped, image[1:5, 2:6])
        cropped[0, 0] = -1
        assert image[1, 2] != -1

    def test_identity_alignment_preserves_pixels(self):
        image = np.arange(25, dtype=float).reshape(5, 5)
        alignment = Alignment(2, 2, 1, 0, output_size_px=5, output_solar_radius_px=1)
        np.testing.assert_allclose(align_north_up(image, alignment, order=0), image)

    @pytest.mark.parametrize(
        ("rotation_deg", "expected_row"),
        [(90, 1), (-90, 3)],
    )
    def test_rotation_sign_moves_source_right_to_expected_cardinal(
        self, rotation_deg, expected_row
    ):
        image = np.zeros((5, 5))
        image[2, 3] = 1.0  # Marker eastward in image coordinates: source-right.
        alignment = Alignment(
            2, 2, 1, rotation_deg, output_size_px=5, output_solar_radius_px=1
        )
        transformed = align_north_up(image, alignment, order=0, cval=0)
        assert transformed[expected_row, 2] == 1.0
        assert transformed.sum() == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("center_x_px", np.nan),
            ("center_y_px", np.inf),
            ("solar_radius_px", True),
            ("rotation_deg", False),
            ("rotation_deg", 181),
            ("output_solar_radius_px", -1),
            ("output_solar_radius_px", 3),
            ("output_size_px", True),
            ("output_size_px", 0),
        ],
    )
    def test_alignment_rejects_invalid_numbers(self, field, value):
        values = {
            "center_x_px": 2,
            "center_y_px": 2,
            "solar_radius_px": 1,
            "rotation_deg": 0,
            "output_size_px": 5,
            "output_solar_radius_px": 1,
        }
        values[field] = value
        with pytest.raises(ValueError):
            Alignment(**values)

    def test_reflection_requires_manifest_evidence(self, tmp_path):
        artifact = tmp_path / "observation.bin"
        artifact.write_bytes(b"observation")
        manifest = _observation_manifest(tmp_path, artifact)
        manifest["alignment"]["reflected"] = True
        manifest["alignment"]["parity_evidence"] = ""
        with pytest.raises(ValueError, match="parity evidence"):
            validate_observation_manifest(manifest, tmp_path)


class TestProfiles:
    def test_sampling_is_invariant_to_gain_and_offset(self):
        size, center, solar_radius = 161, 80.0, 30.0
        y, x = np.indices((size, size))
        pa = position_angle_deg(x, y, center_x=center, center_y=center)
        image = 3 + np.cos(np.radians(pa - 45))
        first = sample_angular_profile(
            image, center_x_px=center, center_y_px=center,
            solar_radius_px=solar_radius, radius_rsun=2.0,
        )
        second = sample_angular_profile(
            image * 7 + 19, center_x_px=center, center_y_px=center,
            solar_radius_px=solar_radius, radius_rsun=2.0,
        )
        assert first.status == second.status == "ok"
        np.testing.assert_allclose(first.values, second.values, atol=1e-10)

    def test_low_coverage_is_not_scored(self):
        image = np.ones((101, 101))
        mask = np.zeros_like(image, dtype=bool)
        mask[:, :50] = True
        profile = sample_angular_profile(
            image, center_x_px=50, center_y_px=50, solar_radius_px=20,
            radius_rsun=2.0, mask=mask,
        )
        assert profile.status == "not_enough_coverage"
        assert profile.coverage < 0.8
        assert profile.mad is None

    def test_constant_profile_is_degenerate(self):
        profile = sample_angular_profile(
            np.ones((101, 101)), center_x_px=50, center_y_px=50,
            solar_radius_px=20, radius_rsun=2.0,
        )
        assert profile.status == "degenerate"

    def test_peak_across_seam_is_detected_once(self):
        peaks = detect_streamer_peaks(_profile((359.0, 120.0)))
        assert len(peaks) == 2
        assert np.min(circular_distance_deg(peaks, 359.0)) <= 2
        assert np.min(circular_distance_deg(peaks, 120.0)) <= 2


class TestMetrics:
    def test_assignment_is_optimal_and_circular(self):
        matches = match_peaks(np.array([359.0, 100.0]), np.array([1.0, 110.0]))
        assert sorted(match["error_deg"] for match in matches) == [2.0, 10.0]

    def test_missing_and_extra_peaks_reduce_precision_or_recall(self):
        extra_prediction = score_profile_pair(_profile((20, 100)), _profile((22,)))
        missing_prediction = score_profile_pair(_profile((20,)), _profile((22, 100)))
        assert extra_prediction["precision_at_10deg"] == pytest.approx(0.5)
        assert extra_prediction["recall_at_10deg"] == pytest.approx(1.0)
        assert missing_prediction["precision_at_10deg"] == pytest.approx(1.0)
        assert missing_prediction["recall_at_10deg"] == pytest.approx(0.5)

    def test_east_west_delta_uses_observed_hemisphere(self):
        result = score_profile_pair(_profile((20, 220)), _profile((28, 222)))
        assert result["east_error_deg"] == pytest.approx(8, abs=1)
        assert result["west_error_deg"] == pytest.approx(2, abs=1)
        assert result["east_minus_west_error_deg"] > 0

    def test_unavailable_profile_does_not_invent_zeros(self):
        unavailable = AngularProfile(2.0, np.arange(360.0), np.full(360, np.nan), 0.4, None, None, "not_enough_coverage")
        result = score_profile_pair(_profile((20,)), unavailable)
        assert result["status"] == "observation_not_enough_coverage"
        assert result["streamer_pa_mae_deg"] is None
        assert result["angular_profile_correlation"] is None


def _observation_manifest(root: Path, artifact: Path) -> dict:
    return {
        "schema_version": 1,
        "role": "primary",
        "selection_order": 1,
        "selected_at": "2026-08-12T21:00:00+02:00",
        "acquired_at": "2026-08-12T18:31:00Z",
        "source": {
            "provider": "test",
            "instrument": "test camera",
            "public_url": "https://example.test/image",
        },
        "artifact": {
            "path": str(artifact.relative_to(root)),
            "sha256": sha256_file(artifact),
            "media_type": "image/png",
        },
        "alternatives_consulted_before_lock": False,
        "alignment": {
            "center_x_px": 10,
            "center_y_px": 10,
            "solar_radius_px": 5,
            "rotation_deg": 0,
            "rotation_source": "wcs",
            "reflected": False,
            "parity_evidence": None,
            "optimized_against_prediction": False,
        },
    }


class TestProvenance:
    def test_frozen_prediction_loads_from_git_and_has_exact_crop(self):
        manifest = load_json(ROOT / "docs/manifests/frozen-prediction.json")
        validate_prediction_manifest(manifest)
        image = load_frozen_prediction(manifest, ROOT)
        assert image.shape == (789, 789, 4)
        np.testing.assert_allclose(
            image[0, 0], [23 / 255, 18 / 255, 14 / 255, 1.0], atol=1e-7
        )

    def test_observation_hash_is_enforced(self, tmp_path):
        artifact = tmp_path / "observation.bin"
        artifact.write_bytes(b"observation")
        manifest = _observation_manifest(tmp_path, artifact)
        validate_observation_manifest(manifest, tmp_path)
        artifact.write_bytes(b"changed")
        with pytest.raises(ValueError, match="path/hash"):
            validate_observation_manifest(manifest, tmp_path)

    def test_optimized_flag_is_mandatory(self, tmp_path):
        artifact = tmp_path / "observation.bin"
        artifact.write_bytes(b"observation")
        manifest = _observation_manifest(tmp_path, artifact)
        del manifest["alignment"]["optimized_against_prediction"]
        with pytest.raises(ValueError, match="optimized_against_prediction"):
            validate_observation_manifest(manifest, tmp_path)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("selection_order", 2, "before consulting alternatives"),
            ("alternatives_consulted_before_lock", True, "before consulting alternatives"),
        ],
    )
    def test_primary_selection_guard(self, tmp_path, field, value, message):
        artifact = tmp_path / "observation.bin"
        artifact.write_bytes(b"observation")
        manifest = _observation_manifest(tmp_path, artifact)
        manifest[field] = value
        with pytest.raises(ValueError, match=message):
            validate_observation_manifest(manifest, tmp_path)

    def test_alignment_optimization_is_rejected(self, tmp_path):
        artifact = tmp_path / "observation.bin"
        artifact.write_bytes(b"observation")
        manifest = _observation_manifest(tmp_path, artifact)
        manifest["alignment"]["optimized_against_prediction"] = True
        with pytest.raises(ValueError, match="optimized"):
            validate_observation_manifest(manifest, tmp_path)

    @pytest.mark.parametrize("value", [True, 1, 0, None, "false"])
    def test_alignment_optimization_must_be_exact_false(self, tmp_path, value):
        artifact = tmp_path / "observation.bin"
        artifact.write_bytes(b"observation")
        manifest = _observation_manifest(tmp_path, artifact)
        manifest["alignment"]["optimized_against_prediction"] = value
        with pytest.raises(ValueError, match="exactly false"):
            validate_observation_manifest(manifest, tmp_path)

    def test_artifact_parent_escape_is_rejected(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"observation")
        manifest = _observation_manifest(tmp_path, outside)
        manifest["artifact"]["path"] = "../outside.bin"
        with pytest.raises(ValueError, match="inside the repository"):
            validate_observation_manifest(manifest, root)

    def test_artifact_symlink_escape_is_rejected(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"observation")
        link = root / "observation.bin"
        link.symlink_to(outside)
        manifest = _observation_manifest(root, link)
        with pytest.raises(ValueError, match="inside the repository"):
            validate_observation_manifest(manifest, root)

    def test_mask_parent_escape_is_rejected(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        artifact = root / "observation.bin"
        artifact.write_bytes(b"observation")
        outside = tmp_path / "mask.bin"
        outside.write_bytes(b"mask")
        manifest = _observation_manifest(root, artifact)
        manifest["mask"] = {
            "path": "../mask.bin",
            "sha256": sha256_file(outside),
            "valid_when": "nonzero",
        }
        with pytest.raises(ValueError, match="inside the repository"):
            validate_observation_manifest(manifest, root)

    def test_mask_hash_is_enforced(self, tmp_path):
        artifact = tmp_path / "observation.bin"
        artifact.write_bytes(b"observation")
        mask = tmp_path / "mask.bin"
        mask.write_bytes(b"mask")
        manifest = _observation_manifest(tmp_path, artifact)
        manifest["mask"] = {
            "path": "mask.bin",
            "sha256": sha256_file(mask),
            "valid_when": "nonzero",
        }
        validate_observation_manifest(manifest, tmp_path)
        manifest["mask"]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="mask path/hash"):
            validate_observation_manifest(manifest, tmp_path)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("center_x_px", np.nan),
            ("center_y_px", np.inf),
            ("solar_radius_px", True),
            ("solar_radius_px", 0),
            ("rotation_deg", False),
            ("rotation_deg", -np.inf),
            ("rotation_deg", 181),
        ],
    )
    def test_manifest_rejects_invalid_alignment_numbers(self, tmp_path, field, value):
        artifact = tmp_path / "observation.bin"
        artifact.write_bytes(b"observation")
        manifest = _observation_manifest(tmp_path, artifact)
        manifest["alignment"][field] = value
        with pytest.raises(ValueError):
            validate_observation_manifest(manifest, tmp_path)

    def test_center_must_be_inside_decoded_image(self, tmp_path):
        artifact = tmp_path / "observation.bin"
        artifact.write_bytes(b"observation")
        manifest = _observation_manifest(tmp_path, artifact)
        manifest["alignment"]["center_x_px"] = 10
        with pytest.raises(ValueError, match="inside the observation image"):
            validate_alignment_for_image(manifest, (10, 10, 3))

    def test_official_score_rejects_parameter_drift(self):
        score = {
            "schema_version": 1,
            "analysis_kind": "preregistered",
            "protocol_lock": {},
            "prediction": {},
            "observation": {},
            "parameters": {"rotation_optimization": True},
            "results": {},
            "confirmatory": {"hypothesis": "east_error_gt_west_error"},
        }
        with pytest.raises(ValueError):
            validate_official_score(score)

    def test_repository_protocol_lock_is_valid(self):
        lock = load_json(ROOT / "docs/manifests/validation-protocol-lock.json")
        validate_protocol_lock(lock, ROOT)
        assert load_canonical_protocol_lock(ROOT) == lock

    def test_alternative_protocol_lock_is_rejected(self, tmp_path):
        alternative = tmp_path / "alternative-lock.json"
        alternative.write_text(
            (ROOT / "docs/manifests/validation-protocol-lock.json").read_text()
        )
        with pytest.raises(ValueError, match="canonical protocol lock"):
            load_canonical_protocol_lock(ROOT, alternative)


class TestCli:
    def test_missing_manifest_has_short_error_without_traceback(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/phase_g.py"),
                "--observation-manifest",
                "outputs/does-not-exist.manifest.json",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "phase_g: error:" in result.stderr
        assert "Traceback" not in result.stderr
        assert not (ROOT / "outputs/preregistered_score.json").exists()

    def test_protocol_lock_override_is_not_a_cli_option(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/phase_g.py"),
                "--observation-manifest",
                "outputs/does-not-exist.manifest.json",
                "--protocol-lock",
                "alternative.json",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "unrecognized arguments: --protocol-lock" in result.stderr
        assert "Traceback" not in result.stderr
