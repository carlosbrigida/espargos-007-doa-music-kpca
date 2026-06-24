from pathlib import Path
import sys
import json
import time

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.cache_dataset import DatasetCache
from src.espargos_geometry import RECEIVER_ARRAYS, true_azimuth_deg
from src.metrics import angular_error_deg


SCENARIOS = {
    "standing_center": "espargos_007_human_helmet_standing_center_1.npz",
    "meanders_nw_se": "espargos_007_human_helmet_meanders_nw_se_1.npz",
    "meanders_sw_ne": "espargos_007_human_helmet_meanders_sw_ne_1.npz",
}

WINDOW_SIZE = 100
STRIDE = 50


def get_unitary_rootmusic_estimator(chunksize=4, shed_coeff_ratio=0):
    I = np.eye(chunksize // 2)
    J = np.flip(np.eye(chunksize // 2), axis=-1)
    Q = np.asmatrix(np.block([[I, 1.0j * I], [J, -1.0j * J]]) / np.sqrt(2))

    def unitary_rootmusic(R):
        C = np.real(Q.H @ R @ Q)

        eig_val, eig_vec = np.linalg.eigh(C)
        eig_vec = eig_vec[:, ::-1]

        source_count = 1
        En = eig_vec[:, source_count:]

        ENSQ = Q @ En @ En.T @ Q.H

        coeffs = np.asarray(
            [np.trace(ENSQ, offset=diag) for diag in range(1, len(R))]
        )

        coeffs = coeffs[: int(len(coeffs) * (1 - shed_coeff_ratio))]
        coeffs = np.hstack((coeffs[::-1], np.trace(ENSQ), coeffs.conj()))

        roots = np.roots(coeffs)
        roots = roots[np.abs(roots) < 1.0]

        if roots.size == 0:
            return np.nan, np.nan

        largest_root = np.argmax(1 / (1.0 - np.abs(roots)))
        electrical_angle = np.angle(roots[largest_root])
        physical_angle = np.arcsin(np.clip(electrical_angle / np.pi, -1.0, 1.0))

        return float(physical_angle), float(np.abs(roots[largest_root]))

    return unitary_rootmusic


def covariance_per_array_luis(csi):
    return np.einsum("dbrms,dbrns->bmn", csi, np.conj(csi)) / csi.shape[0]


def iter_windows(csi, pos):
    for start in range(0, csi.shape[0] - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        yield csi[start:end], np.mean(pos[start:end], axis=0)


def evaluate_scenario(scenario_name, filename):
    cache_file = PROJECT_DIR / "data" / "cache" / filename

    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}")

    data = DatasetCache(str(cache_file)).load()

    csi = data["csi"]
    pos = data["pos"]

    estimator = get_unitary_rootmusic_estimator(4)

    errors_by_array = {i: [] for i in range(4)}

    for csi_window, mean_position in iter_windows(csi, pos):
        R_all = covariance_per_array_luis(csi_window)

        for array_index in range(4):
            array = RECEIVER_ARRAYS[array_index]
            true_angle = true_azimuth_deg(array, mean_position)

            estimated_angle, power = estimator(R_all[array_index])

            if np.isnan(estimated_angle):
                continue

            error = angular_error_deg(
                np.rad2deg(estimated_angle),
                true_angle,
            )

            errors_by_array[array_index].append(error)

    mae_by_array = {
        array_index: float(np.mean(errors)) if len(errors) > 0 else np.nan
        for array_index, errors in errors_by_array.items()
    }

    all_errors = [
        error
        for errors in errors_by_array.values()
        for error in errors
    ]

    return {
        "scenario": scenario_name,
        "snapshots": int(csi.shape[0]),
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "windows": int((csi.shape[0] - WINDOW_SIZE) // STRIDE + 1),
        "mae_by_array": mae_by_array,
        "mae_global": float(np.mean(all_errors)) if len(all_errors) > 0 else np.nan,
    }


def main():
    print("=" * 80)
    print("LUIS uROOT-MUSIC BASELINE - CURRENT CACHE")
    print("=" * 80)
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Stride: {STRIDE}")
    print("CRAP: disabled in current cache pipeline")
    print()

    start_time = time.perf_counter()

    results = []

    for scenario_name, filename in SCENARIOS.items():
        result = evaluate_scenario(scenario_name, filename)
        results.append(result)

        print(f"\nScenario: {scenario_name}")
        print(f"Snapshots: {result['snapshots']}")
        print(f"Windows: {result['windows']}")
        print(f"MAE global: {result['mae_global']:.2f}°")

        for array_index, mae in result["mae_by_array"].items():
            print(f"  Array {array_index}: {mae:.2f}°")

    elapsed = time.perf_counter() - start_time

    output = {
        "experiment": "luis_uroot_music_baseline_current_cache",
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "crap": False,
        "elapsed_seconds": elapsed,
        "results": results,
    }

    output_file = (
        PROJECT_DIR
        / "outputs"
        / "metrics"
        / "luis_uroot_music_baseline_current_cache.json"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Elapsed: {elapsed:.2f} s")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()