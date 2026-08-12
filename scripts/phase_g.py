"""Evalúa una observación bloqueada contra la predicción congelada de corona26.

El modo oficial exige un protocol lock y un observation manifest válidos. No
descarga observaciones ni ejecuta el pipeline físico.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from matplotlib import image as mpl_image

from corona26.validation.alignment import align_north_up
from corona26.validation.metrics import score_profiles
from corona26.validation.profiles import sample_angular_profile
from corona26.validation.provenance import (
    alignment_from_manifest,
    confined_file,
    load_frozen_prediction,
    load_json,
    load_canonical_protocol_lock,
    sha256_file,
    validate_alignment_for_image,
    validate_observation_manifest,
    validate_official_score,
    validate_prediction_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
RADII = (1.5, 2.0, 2.5)
PARAMETERS = {
    "radii_rsun": list(RADII),
    "annulus_half_width_rsun": 0.05,
    "minimum_coverage": 0.8,
    "smoothing_sigma_deg": 3.0,
    "peak_prominence_mad": 0.5,
    "minimum_peak_distance_deg": 15.0,
    "match_tolerance_deg": 10.0,
    "rotation_optimization": False,
}


def _profiles(image: np.ndarray, center: float, solar_radius: float, mask=None):
    return {
        radius: sample_angular_profile(
            image,
            center_x_px=center,
            center_y_px=center,
            solar_radius_px=solar_radius,
            radius_rsun=radius,
            mask=mask,
        )
        for radius in RADII
    }


def run_official(
    observation_manifest_path: Path,
    output_path: Path,
) -> dict:
    """Run the locked score, failing before output on any provenance violation."""
    if output_path.name != "preregistered_score.json":
        raise ValueError("official output must be named preregistered_score.json")
    protocol_lock_path = ROOT / "docs/manifests/validation-protocol-lock.json"
    lock = load_canonical_protocol_lock(ROOT)
    prediction_manifest_path = confined_file(
        ROOT,
        lock["prediction_manifest"]["path"],
        "prediction manifest path",
    )
    prediction_manifest = load_json(prediction_manifest_path)
    validate_prediction_manifest(prediction_manifest)

    observation_manifest = load_json(observation_manifest_path)
    validate_observation_manifest(observation_manifest, ROOT)
    selected_at = datetime.fromisoformat(
        observation_manifest["selected_at"].replace("Z", "+00:00")
    )
    locked_at = datetime.fromisoformat(lock["locked_at"].replace("Z", "+00:00"))
    if selected_at < locked_at:
        raise ValueError("primary observation must be selected after the protocol lock")
    prediction = load_frozen_prediction(prediction_manifest, ROOT)

    observation_path = confined_file(
        ROOT, observation_manifest["artifact"]["path"], "observation artifact path"
    )
    observation_raw = mpl_image.imread(observation_path)
    validate_alignment_for_image(observation_manifest, observation_raw.shape)
    alignment = alignment_from_manifest(observation_manifest)
    observation = align_north_up(observation_raw, alignment)
    observation_mask = np.isfinite(observation[..., 0] if observation.ndim == 3 else observation)
    if "mask" in observation_manifest:
        mask_path = confined_file(
            ROOT, observation_manifest["mask"]["path"], "observation mask path"
        )
        raw_mask = mpl_image.imread(mask_path)
        if raw_mask.shape[:2] != observation_raw.shape[:2]:
            raise ValueError("observation mask dimensions must match the observation")
        if raw_mask.ndim == 3:
            raw_mask = raw_mask[..., 0]
        transformed_mask = align_north_up(
            (raw_mask != 0).astype(float), alignment, order=0, cval=0.0
        )
        observation_mask &= transformed_mask > 0.5

    geometry = prediction_manifest["geometry"]
    prediction_profiles = _profiles(
        prediction,
        float(geometry["center_x_px"]),
        float(geometry["solar_radius_px"]),
    )
    output_center = (alignment.output_size_px - 1) / 2.0
    observation_profiles = _profiles(
        observation,
        output_center,
        alignment.output_solar_radius_px,
        observation_mask,
    )
    results = score_profiles(prediction_profiles, observation_profiles)
    delta = results["macro"]["east_minus_west_error_deg"]
    score = {
        "schema_version": 1,
        "analysis_kind": "preregistered",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_lock": {
            "path": str(protocol_lock_path.relative_to(ROOT)),
            "sha256": sha256_file(protocol_lock_path),
        },
        "prediction": {
            "manifest_path": str(prediction_manifest_path.relative_to(ROOT)),
            "artifact_sha256": prediction_manifest["artifact"]["sha256"],
            "git_commit": prediction_manifest["git_commit"],
        },
        "observation": {
            "manifest_path": str(observation_manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256_file(observation_manifest_path),
            "artifact_sha256": observation_manifest["artifact"]["sha256"],
        },
        "parameters": PARAMETERS,
        "results": results,
        "confirmatory": {
            "hypothesis": "east_error_gt_west_error",
            "east_minus_west_error_deg": delta,
            "supported": delta > 0 if delta is not None else None,
        },
    }
    validate_official_score(score)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(score, indent=2, allow_nan=False) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
        os.replace(temporary_name, output_path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/preregistered_score.json",
    )
    args = parser.parse_args()
    try:
        paths = []
        for path in (args.observation_manifest, args.output):
            resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
            if resolved == ROOT or ROOT not in resolved.parents:
                raise ValueError("manifest and output paths must stay inside the repository")
            paths.append(resolved)
        observation_manifest, output = paths
        run_official(observation_manifest, output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"phase_g: error: {exc}\n")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
