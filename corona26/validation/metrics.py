"""Preregistered circular metrics for angular corona profiles."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from corona26.validation.alignment import circular_distance_deg
from corona26.validation.profiles import (
    AngularProfile,
    detect_streamer_peaks,
    smooth_circular_profile,
)


def match_peaks(predicted_pa: np.ndarray, observed_pa: np.ndarray) -> list[dict[str, float]]:
    """Return the globally minimum-cost one-to-one circular assignment."""
    predicted = np.asarray(predicted_pa, dtype=float)
    observed = np.asarray(observed_pa, dtype=float)
    if predicted.size == 0 or observed.size == 0:
        return []
    costs = circular_distance_deg(predicted[:, None], observed[None, :])
    pred_indices, obs_indices = linear_sum_assignment(costs)
    return [
        {
            "predicted_pa_deg": float(predicted[p]),
            "observed_pa_deg": float(observed[o]),
            "error_deg": float(costs[p, o]),
        }
        for p, o in zip(pred_indices, obs_indices, strict=True)
    ]


def profile_correlation(predicted: AngularProfile, observed: AngularProfile) -> float | None:
    """Correlate profiles at the registered orientation, without rotation search."""
    predicted_values = smooth_circular_profile(predicted.values)
    observed_values = smooth_circular_profile(observed.values)
    valid = np.isfinite(predicted_values) & np.isfinite(observed_values)
    if valid.sum() < 3:
        return None
    a, b = predicted_values[valid], observed_values[valid]
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _side_summary(matches: list[dict[str, float]], start: float, stop: float) -> float | None:
    errors = [
        match["error_deg"]
        for match in matches
        if start <= match["observed_pa_deg"] < stop
    ]
    return float(np.mean(errors)) if errors else None


def score_profile_pair(
    predicted: AngularProfile,
    observed: AngularProfile,
    *,
    tolerance_deg: float = 10.0,
) -> dict[str, Any]:
    """Score one radius while preserving unavailable and degenerate states."""
    if predicted.status != "ok" or observed.status != "ok":
        return {
            "status": f"prediction_{predicted.status}" if predicted.status != "ok" else f"observation_{observed.status}",
            "prediction_coverage": predicted.coverage,
            "observation_coverage": observed.coverage,
            "streamer_pa_mae_deg": None,
            "precision_at_10deg": None,
            "recall_at_10deg": None,
            "angular_profile_correlation": None,
            "east_error_deg": None,
            "west_error_deg": None,
            "east_minus_west_error_deg": None,
        }

    predicted_peaks = detect_streamer_peaks(predicted)
    observed_peaks = detect_streamer_peaks(observed)
    matches = match_peaks(predicted_peaks, observed_peaks)
    accepted = [match for match in matches if match["error_deg"] <= tolerance_deg]
    mae = float(np.mean([match["error_deg"] for match in matches])) if matches else None
    precision = len(accepted) / predicted_peaks.size if predicted_peaks.size else None
    recall = len(accepted) / observed_peaks.size if observed_peaks.size else None
    east = _side_summary(matches, 0.0, 180.0)
    west = _side_summary(matches, 180.0, 360.0)
    status = "ok" if predicted_peaks.size and observed_peaks.size else "no_peaks"
    return {
        "status": status,
        "prediction_coverage": predicted.coverage,
        "observation_coverage": observed.coverage,
        "predicted_peak_pa_deg": predicted_peaks.tolist(),
        "observed_peak_pa_deg": observed_peaks.tolist(),
        "matches": matches,
        "streamer_pa_mae_deg": mae,
        "precision_at_10deg": float(precision) if precision is not None else None,
        "recall_at_10deg": float(recall) if recall is not None else None,
        "angular_profile_correlation": profile_correlation(predicted, observed),
        "east_error_deg": east,
        "west_error_deg": west,
        "east_minus_west_error_deg": east - west if east is not None and west is not None else None,
    }


def _mean_available(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def score_profiles(
    predicted: dict[float, AngularProfile],
    observed: dict[float, AngularProfile],
) -> dict[str, Any]:
    """Score all registered radii and compute a macro mean over available values."""
    if predicted.keys() != observed.keys():
        raise ValueError("prediction and observation radii must match")
    by_radius = {
        str(radius): score_profile_pair(predicted[radius], observed[radius])
        for radius in sorted(predicted)
    }
    rows = list(by_radius.values())
    keys = (
        "streamer_pa_mae_deg",
        "precision_at_10deg",
        "recall_at_10deg",
        "angular_profile_correlation",
        "east_error_deg",
        "west_error_deg",
        "east_minus_west_error_deg",
    )
    return {"by_radius": by_radius, "macro": {key: _mean_available(rows, key) for key in keys}}
