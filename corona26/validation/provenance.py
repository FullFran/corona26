"""Strict provenance and manifest checks for post-eclipse scoring."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from matplotlib import image as mpl_image

from corona26.validation.alignment import Alignment, Crop

SHA256_LENGTH = 64
ALLOWED_ROTATION_SOURCES = {"ephemeris", "wcs", "mount", "astrometry"}
CANONICAL_PROTOCOL_LOCK_PATH = Path("docs/manifests/validation-protocol-lock.json")
# This anchors the lock itself; the lock hashes only the protocol and prediction manifest.
CANONICAL_PROTOCOL_LOCK_SHA256 = "5cff6d9170224394dd181b9137b5f86270eafcea5c51da29965798b1dc7efc6f"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"manifest {path} must contain a JSON object")
    return value


def _require(manifest: dict[str, Any], fields: set[str], name: str) -> None:
    missing = fields - manifest.keys()
    if missing:
        raise ValueError(f"{name} is missing required fields: {', '.join(sorted(missing))}")


def _validate_sha(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != SHA256_LENGTH:
        raise ValueError(f"{name} must be a complete SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc


def _parse_time(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO-8601 timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed


def confined_file(root: Path, value: Any, name: str) -> Path:
    """Resolve an existing relative file strictly inside root.

    Absolute paths, parent traversal, root itself and symlinks escaping root are
    rejected before callers read or hash the file.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must stay inside the repository")
    resolved_root = root.resolve()
    try:
        resolved = (resolved_root / relative).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{name} does not identify an existing file") from exc
    if resolved == resolved_root or resolved_root not in resolved.parents or not resolved.is_file():
        raise ValueError(f"{name} must stay inside the repository")
    return resolved


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def validate_protocol_lock(lock: dict[str, Any], root: Path) -> None:
    _require(lock, {"schema_version", "locked_at", "protocol", "prediction_manifest"}, "protocol lock")
    if lock["schema_version"] != 1:
        raise ValueError("unsupported protocol lock schema")
    _parse_time(lock["locked_at"], "protocol lock locked_at")
    for field in ("protocol", "prediction_manifest"):
        reference = lock[field]
        if not isinstance(reference, dict):
            raise ValueError(f"protocol lock {field} must be an object")
        _require(reference, {"path", "sha256"}, f"protocol lock {field}")
        _validate_sha(reference["sha256"], f"protocol lock {field}.sha256")
        path = confined_file(root, reference["path"], f"protocol lock {field}.path")
        if sha256_file(path) != reference["sha256"]:
            raise ValueError(f"protocol lock hash mismatch for {reference['path']}")


def load_canonical_protocol_lock(root: Path, path: Path | None = None) -> dict[str, Any]:
    """Load the one official lock, anchored independently by path and SHA-256."""
    canonical = confined_file(root, str(CANONICAL_PROTOCOL_LOCK_PATH), "canonical protocol lock")
    requested = canonical if path is None else path.resolve(strict=True)
    if requested != canonical:
        raise ValueError("official scoring requires the canonical protocol lock")
    if sha256_file(canonical) != CANONICAL_PROTOCOL_LOCK_SHA256:
        raise ValueError("canonical protocol lock SHA-256 mismatch")
    lock = load_json(canonical)
    validate_protocol_lock(lock, root)
    return lock


def validate_prediction_manifest(manifest: dict[str, Any]) -> None:
    _require(
        manifest,
        {"schema_version", "artifact", "git_commit", "selector", "crop", "geometry", "transformation"},
        "prediction manifest",
    )
    artifact = manifest["artifact"]
    _require(artifact, {"repository_path", "sha256"}, "prediction artifact")
    _validate_sha(artifact["sha256"], "prediction artifact sha256")
    if manifest["selector"] != "first-panel" or manifest["transformation"] != "explicit_crop_only":
        raise ValueError("official prediction must be the explicitly cropped first panel")
    if not isinstance(manifest["git_commit"], str) or len(manifest["git_commit"]) != 40:
        raise ValueError("prediction git_commit must be a full commit ID")
    bounds = manifest["crop"].get("bounds_px")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 4
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in bounds)
        or not (0 <= bounds[0] < bounds[2] and 0 <= bounds[1] < bounds[3])
    ):
        raise ValueError("prediction crop bounds_px must contain four integers")
    geometry = manifest["geometry"]
    _require(geometry, {"center_x_px", "center_y_px", "solar_radius_px", "orientation"}, "prediction geometry")
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    center_x = _finite_number(geometry["center_x_px"], "prediction center_x_px")
    center_y = _finite_number(geometry["center_y_px"], "prediction center_y_px")
    _finite_number(geometry["solar_radius_px"], "prediction solar_radius_px", positive=True)
    if not (0 <= center_x < width and 0 <= center_y < height):
        raise ValueError("prediction center must be inside the cropped image")
    if geometry["orientation"] != "solar_north_up":
        raise ValueError("official prediction must be solar-north-up")


def validate_observation_manifest(manifest: dict[str, Any], root: Path) -> None:
    _require(
        manifest,
        {
            "schema_version", "role", "selection_order", "selected_at", "acquired_at",
            "source", "artifact", "alternatives_consulted_before_lock", "alignment",
        },
        "observation manifest",
    )
    if manifest["schema_version"] != 1 or manifest["role"] != "primary":
        raise ValueError("official observation must be a schema-1 primary observation")
    if manifest["selection_order"] != 1 or manifest["alternatives_consulted_before_lock"] is not False:
        raise ValueError("primary observation must be locked before consulting alternatives")
    selected_at = _parse_time(manifest["selected_at"], "observation selected_at")
    acquired_at = _parse_time(manifest["acquired_at"], "observation acquired_at")
    if selected_at < acquired_at:
        raise ValueError("observation cannot be selected before it was acquired")
    _require(
        manifest["source"],
        {"provider", "instrument", "public_url"},
        "observation source",
    )
    artifact = manifest["artifact"]
    _require(artifact, {"path", "sha256", "media_type"}, "observation artifact")
    _validate_sha(artifact["sha256"], "observation artifact sha256")
    path = confined_file(root, artifact["path"], "observation artifact path")
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError("observation artifact path/hash is invalid")
    alignment = manifest["alignment"]
    _require(
        alignment,
        {
            "center_x_px", "center_y_px", "solar_radius_px", "rotation_deg",
            "rotation_source", "reflected", "parity_evidence",
            "optimized_against_prediction",
        },
        "observation alignment",
    )
    _finite_number(alignment["center_x_px"], "alignment center_x_px")
    _finite_number(alignment["center_y_px"], "alignment center_y_px")
    _finite_number(alignment["solar_radius_px"], "alignment solar_radius_px", positive=True)
    rotation = _finite_number(alignment["rotation_deg"], "alignment rotation_deg")
    if not -180.0 <= rotation <= 180.0:
        raise ValueError("alignment rotation_deg must be within [-180, 180]")
    if not isinstance(alignment["reflected"], bool):
        raise ValueError("alignment reflected must be boolean")
    if alignment["rotation_source"] not in ALLOWED_ROTATION_SOURCES:
        raise ValueError("rotation must come from ephemeris, WCS, mount or astrometry")
    if alignment["reflected"] and not alignment["parity_evidence"]:
        raise ValueError("reflection requires documented parity evidence")
    if alignment["optimized_against_prediction"] is not False:
        raise ValueError("optimized_against_prediction must be exactly false")
    if "mask" in manifest:
        mask = manifest["mask"]
        _require(mask, {"path", "sha256", "valid_when"}, "observation mask")
        _validate_sha(mask["sha256"], "observation mask sha256")
        mask_path = confined_file(root, mask["path"], "observation mask path")
        if (
            mask["valid_when"] != "nonzero"
            or sha256_file(mask_path) != mask["sha256"]
        ):
            raise ValueError("observation mask path/hash/convention is invalid")


def validate_alignment_for_image(manifest: dict[str, Any], image_shape: tuple[int, ...]) -> None:
    """Check that the declared source center lies inside the decoded image."""
    if len(image_shape) < 2 or image_shape[0] <= 0 or image_shape[1] <= 0:
        raise ValueError("observation image has invalid dimensions")
    alignment = manifest["alignment"]
    center_x = float(alignment["center_x_px"])
    center_y = float(alignment["center_y_px"])
    if not (0 <= center_x < image_shape[1] and 0 <= center_y < image_shape[0]):
        raise ValueError("alignment center must be inside the observation image")


def load_frozen_prediction(manifest: dict[str, Any], root: Path):
    """Read the exact committed PNG blob and apply its explicit panel crop."""
    validate_prediction_manifest(manifest)
    artifact = manifest["artifact"]
    spec = f"{manifest['git_commit']}:{artifact['repository_path']}"
    try:
        data = subprocess.run(
            ["git", "show", spec], cwd=root, check=True, capture_output=True
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"cannot load frozen prediction {spec}") from exc
    if sha256_bytes(data) != artifact["sha256"]:
        raise ValueError("frozen prediction blob does not match its SHA-256")
    import io

    image = mpl_image.imread(io.BytesIO(data), format="png")
    return Crop(*manifest["crop"]["bounds_px"]).apply(image)


def alignment_from_manifest(manifest: dict[str, Any]) -> Alignment:
    values = manifest["alignment"]
    return Alignment(
        center_x_px=float(values["center_x_px"]),
        center_y_px=float(values["center_y_px"]),
        solar_radius_px=float(values["solar_radius_px"]),
        rotation_deg=float(values["rotation_deg"]),
        reflected=bool(values["reflected"]),
    )


def validate_official_score(score: dict[str, Any]) -> None:
    """Reject incomplete, exploratory or hindsight-optimized official scores."""
    _require(
        score,
        {"schema_version", "analysis_kind", "protocol_lock", "prediction", "observation", "parameters", "results", "confirmatory"},
        "official score",
    )
    if score["schema_version"] != 1 or score["analysis_kind"] != "preregistered":
        raise ValueError("official score must be schema-1 preregistered output")
    _require(score["protocol_lock"], {"path", "sha256"}, "score protocol lock")
    if score["protocol_lock"] != {
        "path": str(CANONICAL_PROTOCOL_LOCK_PATH),
        "sha256": CANONICAL_PROTOCOL_LOCK_SHA256,
    }:
        raise ValueError("official score must reference the canonical protocol lock")
    _require(score["prediction"], {"manifest_path", "artifact_sha256", "git_commit"}, "score prediction")
    _require(score["observation"], {"manifest_path", "manifest_sha256", "artifact_sha256"}, "score observation")
    for owner, field in (
        (score["protocol_lock"], "sha256"),
        (score["prediction"], "artifact_sha256"),
        (score["observation"], "manifest_sha256"),
        (score["observation"], "artifact_sha256"),
    ):
        _validate_sha(owner[field], f"official score {field}")
    _require(score["results"], {"by_radius", "macro"}, "official score results")
    if set(score["results"]["by_radius"]) != {"1.5", "2.0", "2.5"}:
        raise ValueError("official score must contain exactly the three locked radii")
    metric_fields = {
        "status", "prediction_coverage", "observation_coverage",
        "streamer_pa_mae_deg", "precision_at_10deg", "recall_at_10deg",
        "angular_profile_correlation", "east_error_deg", "west_error_deg",
        "east_minus_west_error_deg",
    }
    for radius, result in score["results"]["by_radius"].items():
        _require(result, metric_fields, f"official score radius {radius}")
    _require(score["results"]["macro"], metric_fields - {
        "status", "prediction_coverage", "observation_coverage"
    }, "official score macro")
    if score["parameters"] != {
        "radii_rsun": [1.5, 2.0, 2.5],
        "annulus_half_width_rsun": 0.05,
        "minimum_coverage": 0.8,
        "smoothing_sigma_deg": 3.0,
        "peak_prominence_mad": 0.5,
        "minimum_peak_distance_deg": 15.0,
        "match_tolerance_deg": 10.0,
        "rotation_optimization": False,
    }:
        raise ValueError("official scoring parameters differ from the protocol lock")
    _require(
        score["confirmatory"],
        {"hypothesis", "east_minus_west_error_deg", "supported"},
        "official score confirmatory result",
    )
    if score["confirmatory"].get("hypothesis") != "east_error_gt_west_error":
        raise ValueError("official score must report the locked east/west hypothesis")
