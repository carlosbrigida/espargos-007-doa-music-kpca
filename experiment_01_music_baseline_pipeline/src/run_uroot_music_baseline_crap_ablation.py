from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


CACHE_DIR = Path("data/cache")
OUTPUT_DIR = Path("outputs/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 100
STRIDE = 50
VALIDATION_FRACTION = 0.2

SCENARIOS = {
    "standing_center": "espargos_007_human_helmet_standing_center_1.npz",
    "meanders_nw_se": "espargos_007_human_helmet_meanders_nw_se_1.npz",
    "meanders_sw_ne": "espargos_007_human_helmet_meanders_sw_ne_1.npz",
}


def get_unitary_rootmusic_estimator(chunksize: int = 4):
    eye = np.eye(chunksize // 2)
    exchange = np.flip(np.eye(chunksize // 2), axis=-1)

    q_matrix = np.asmatrix(
        np.block([[eye, 1.0j * eye], [exchange, -1.0j * exchange]])
        / np.sqrt(2)
    )

    def unitary_rootmusic(covariance: np.ndarray) -> float:
        transformed = np.real(q_matrix.H @ covariance @ q_matrix)

        eig_vals, eig_vecs = np.linalg.eigh(transformed)
        eig_vecs = eig_vecs[:, ::-1]

        source_count = 1
        noise_subspace = eig_vecs[:, source_count:]

        ensq = q_matrix @ noise_subspace @ noise_subspace.T @ q_matrix.H

        coeffs = np.asarray(
            [np.trace(ensq, offset=diag) for diag in range(1, len(covariance))]
        )

        coeffs = np.hstack((coeffs[::-1], np.trace(ensq), coeffs.conj()))

        roots = np.roots(coeffs)
        roots = roots[np.abs(roots) < 1.0]

        if roots.size == 0:
            return np.nan

        selected_root = roots[np.argmax(1.0 / (1.0 - np.abs(roots)))]

        electrical_angle = np.angle(selected_root)
        physical_angle = np.arcsin(np.clip(electrical_angle / np.pi, -1.0, 1.0))

        return float(physical_angle)

    return unitary_rootmusic


def covariance_per_array_luis(csi: np.ndarray) -> np.ndarray:
    return np.einsum(
        "dbrms,dbrns->bmn",
        csi,
        np.conj(csi),
    ) / csi.shape[0]


def load_geometry():
    import src.espargos_geometry as geometry

    arrays = geometry.RECEIVER_ARRAYS

    array_positions = np.asarray(
        [arrays[i].center for i in sorted(arrays.keys())],
        dtype=float,
    )

    array_rights = np.asarray(
        [arrays[i].right for i in sorted(arrays.keys())],
        dtype=float,
    )

    array_normals = np.asarray(
        [geometry.array_normal(arrays[i]) for i in sorted(arrays.keys())],
        dtype=float,
    )

    return array_positions, array_normals, array_rights

def ground_truth_aoa(position: np.ndarray) -> np.ndarray:
    array_positions, array_normals, array_rights = load_geometry()

    relative_position = position[np.newaxis, :] - array_positions

    normal = np.einsum("ax,ax->a", relative_position, array_normals)
    right = np.einsum("ax,ax->a", relative_position, array_rights)

    return np.arctan2(right, normal)


def angular_error_deg(estimated: float, true: float) -> float:
    error = estimated - true
    error = np.arctan2(np.sin(error), np.cos(error))
    return float(abs(np.rad2deg(error)))


def iter_windows(csi: np.ndarray, pos: np.ndarray):
    for start in range(0, csi.shape[0] - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        yield csi[start:end], np.mean(pos[start:end], axis=0)


def evaluate_scenario(name: str, cache_file: Path, use_crap: bool) -> dict:
    data = np.load(cache_file, allow_pickle=True)

    csi = data["csi"]
    pos = data["pos"]

    split_index = int((1.0 - VALIDATION_FRACTION) * csi.shape[0])

    csi_eval = csi[split_index:]
    pos_eval = pos[split_index:]

    estimator = get_unitary_rootmusic_estimator(4)

    errors_by_array = [[] for _ in range(4)]

    for csi_window, mean_position in iter_windows(csi_eval, pos_eval):
        covariance = covariance_per_array_luis(csi_window)
        true_aoas = ground_truth_aoa(mean_position)

        for array_index in range(4):
            estimated_aoa = estimator(covariance[array_index])

            if np.isnan(estimated_aoa):
                continue

            error = angular_error_deg(estimated_aoa, true_aoas[array_index])
            errors_by_array[array_index].append(error)

    mean_by_array = [
        float(np.mean(errors)) if errors else float("nan")
        for errors in errors_by_array
    ]

    all_errors = [
        error
        for errors in errors_by_array
        for error in errors
    ]

    return {
        "scenario": name,
        "use_crap": use_crap,
        "snapshots_total": int(csi.shape[0]),
        "snapshots_eval": int(csi_eval.shape[0]),
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "mae_by_array_deg": mean_by_array,
        "mae_global_deg": float(np.mean(all_errors)) if all_errors else float("nan"),
        "num_estimates": int(len(all_errors)),
    }


def run(use_crap: bool) -> dict:
    start_time = time.perf_counter()

    results = []

    for scenario_name, filename in SCENARIOS.items():
        cache_file = CACHE_DIR / filename

        print(f"\nScenario: {scenario_name}")
        print(f"Cache: {cache_file}")
        print(f"CRAP: {use_crap}")

        result = evaluate_scenario(
            scenario_name,
            cache_file,
            use_crap=use_crap,
        )

        results.append(result)

        print(f"MAE global: {result['mae_global_deg']:.2f}°")
        print(f"MAE arrays: {result['mae_by_array_deg']}")

    elapsed = time.perf_counter() - start_time

    return {
        "use_crap": use_crap,
        "elapsed_seconds": elapsed,
        "results": results,
    }


def main() -> None:
    print("=" * 80)
    print("uRoot-MUSIC BASELINE - CURRENT PROJECT")
    print("=" * 80)

    # Primeira rodada: sem CRAP
    without_crap = run(use_crap=False)

    output_file = OUTPUT_DIR / "uroot_music_baseline_without_crap.json"
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(without_crap, file, indent=2)

    print(f"\nSaved: {output_file}")

    # A rodada com CRAP será implementada depois de confirmarmos
    # como o CRAP está disponível no projeto atual.
    print("\nRodada com CRAP fica para a próxima etapa.")


if __name__ == "__main__":
    main()