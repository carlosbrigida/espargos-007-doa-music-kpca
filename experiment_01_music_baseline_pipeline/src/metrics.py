import json
from pathlib import Path

import numpy as np


def explained_variance_ratio(eigvals: np.ndarray) -> np.ndarray:
    eigvals = np.real(eigvals)

    total = np.sum(eigvals)

    if total <= 0:
        raise ValueError("Sum of eigenvalues must be positive.")

    return eigvals / total


def cumulative_explained_variance(
    eigvals: np.ndarray,
) -> np.ndarray:
    return np.cumsum(explained_variance_ratio(eigvals))


def n_components_for_variance(
    eigvals: np.ndarray,
    threshold: float,
) -> int:
    cumulative = cumulative_explained_variance(eigvals)

    return int(np.searchsorted(cumulative, threshold) + 1)


def save_json_metrics(
    metrics: dict,
    output_file: str,
) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def angular_error_deg(
    estimated_angle: float,
    true_angle: float,
) -> float:
    """
    Wrapped angular error.

    Returns values between 0 and 180 degrees.
    """
    error = estimated_angle - true_angle

    wrapped_error = (error + 180.0) % 360.0 - 180.0

    return float(abs(wrapped_error))