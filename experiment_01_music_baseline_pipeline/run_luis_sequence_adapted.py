from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.array_geometry import steering_vector_ula, wavelength_from_frequency
from src.cache_dataset import DatasetCache
from src.espargos_geometry import RECEIVER_ARRAYS, true_azimuth_deg
from src.metrics import angular_error_deg, save_json_metrics
from src.music import MusicEstimator


def covariance_per_array_luis(csi: np.ndarray) -> np.ndarray:
    """
    Luis-style covariance.

    CSI shape:
        (snapshots, arrays, rows, cols, subcarriers)

    Output:
        (arrays, cols, cols)
    """
    return np.einsum(
        "dbrms,dbrns->bmn",
        csi,
        np.conj(csi),
    ) / csi.shape[0]


def luis_pca_per_snapshot_array(
    csi: np.ndarray,
    explained_variance: float = 0.90,
) -> tuple[np.ndarray, dict]:
    """
    Adapted from Luis' RAW PCA logic.

    For each snapshot and each array:

        (2, 4, 53)
        -> transpose
        (53, 2, 4)
        -> reshape
        (53, 8)

    Then:
        PCA(real)
        PCA(imag)
        reconstruct
        return to (2, 4, 53)
    """
    if csi.ndim != 5:
        raise ValueError(f"Expected CSI with 5 dims, got {csi.shape}")

    reconstructed = np.empty_like(csi, dtype=np.complex64)

    n_snapshots, n_arrays, n_rows, n_cols, n_subcarriers = csi.shape

    real_components = []
    imag_components = []
    real_variances = []
    imag_variances = []

    for snapshot_i in range(n_snapshots):
        for array_i in range(n_arrays):
            csi_single = csi[snapshot_i, array_i, :, :, :]

            transposed = np.transpose(csi_single, (2, 0, 1))
            matrix_53x8 = transposed.reshape(
                n_subcarriers,
                n_rows * n_cols,
            )

            real_part = matrix_53x8.real
            imag_part = matrix_53x8.imag

            pca_real = PCA(
                n_components=explained_variance,
                svd_solver="full",
            )

            pca_imag = PCA(
                n_components=explained_variance,
                svd_solver="full",
            )

            real_z = pca_real.fit_transform(real_part)
            imag_z = pca_imag.fit_transform(imag_part)

            real_rec = pca_real.inverse_transform(real_z)
            imag_rec = pca_imag.inverse_transform(imag_z)

            reconstructed_matrix = real_rec + 1j * imag_rec

            temp = reconstructed_matrix.reshape(
                n_subcarriers,
                n_rows,
                n_cols,
            )

            reconstructed_single = np.transpose(temp, (1, 2, 0))

            reconstructed[snapshot_i, array_i, :, :, :] = reconstructed_single

            real_components.append(int(pca_real.n_components_))
            imag_components.append(int(pca_imag.n_components_))
            real_variances.append(float(np.sum(pca_real.explained_variance_ratio_)))
            imag_variances.append(float(np.sum(pca_imag.explained_variance_ratio_)))

    stats = {
        "explained_variance_target": float(explained_variance),
        "real_components_mean": float(np.mean(real_components)),
        "real_components_min": int(np.min(real_components)),
        "real_components_max": int(np.max(real_components)),
        "imag_components_mean": float(np.mean(imag_components)),
        "imag_components_min": int(np.min(imag_components)),
        "imag_components_max": int(np.max(imag_components)),
        "real_preserved_variance_mean": float(np.mean(real_variances)),
        "imag_preserved_variance_mean": float(np.mean(imag_variances)),
    }

    return reconstructed, stats


def estimate_music_for_array(
    R: np.ndarray,
    array_index: int,
    mean_position: np.ndarray,
    frequency_hz: float,
    n_sources: int,
) -> dict:
    array = RECEIVER_ARRAYS[array_index]

    true_angle = true_azimuth_deg(array, mean_position)

    wavelength = wavelength_from_frequency(frequency_hz)
    antenna_spacing = wavelength / 2.0

    music = MusicEstimator(n_sources=n_sources)

    eigvals, eigvecs = music.eigendecomposition(R)
    noise_subspace = music.noise_subspace(eigvecs)

    angles_deg = np.linspace(-90.0, 90.0, 721)
    angles_rad = np.deg2rad(angles_deg)

    steering_vectors = np.asarray(
        [
            steering_vector_ula(
                angle_rad=angle_rad,
                n_antennas=R.shape[0],
                antenna_spacing_m=antenna_spacing,
                frequency_hz=frequency_hz,
            )
            for angle_rad in angles_rad
        ]
    )

    spectrum = music.pseudo_spectrum(
        noise_subspace=noise_subspace,
        steering_vectors=steering_vectors,
    )

    estimated_angle = float(angles_deg[np.argmax(spectrum)])
    error = angular_error_deg(estimated_angle, true_angle)

    return {
        "array_index": int(array_index),
        "array_name": array.name,
        "true_aoa_deg": float(true_angle),
        "estimated_aoa_deg": float(estimated_angle),
        "angular_error_deg": float(error),
        "eigenvalues": eigvals.real.tolist(),
    }


def run_music_all_arrays(
    csi: np.ndarray,
    mean_position: np.ndarray,
    frequency_hz: float,
    n_sources: int,
) -> list[dict]:
    R_all = covariance_per_array_luis(csi)

    results = []

    for array_index in range(R_all.shape[0]):
        results.append(
            estimate_music_for_array(
                R=R_all[array_index],
                array_index=array_index,
                mean_position=mean_position,
                frequency_hz=frequency_hz,
                n_sources=n_sources,
            )
        )

    return results


def print_results(title: str, results: list[dict]) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    for item in results:
        print(
            f"Array {item['array_index']} ({item['array_name']}): "
            f"true={item['true_aoa_deg']:.2f}° | "
            f"est={item['estimated_aoa_deg']:.2f}° | "
            f"err={item['angular_error_deg']:.2f}°"
        )


def main() -> None:
    cache_file = (
        PROJECT_DIR
        / "data"
        / "cache"
        / "espargos_007_human_helmet_standing_center_1.npz"
    )

    output_file = (
        PROJECT_DIR
        / "outputs"
        / "metrics"
        / "luis_sequence_adapted_pca_per_snapshot_array.json"
    )

    frequency_hz = 2.472e9
    n_sources = 2
    pca_variance = 0.90

    print()
    print("=" * 70)
    print("LUIS SEQUENCE ADAPTED - PCA PER SNAPSHOT/ARRAY")
    print("=" * 70)
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Cache file: {cache_file}")
    print("Scenario: human_helmet_standing_center_1")
    print("CRAP: disabled for now")
    print(f"PCA variance: {pca_variance}")
    print(f"n_sources: {n_sources}")

    cache = DatasetCache(str(cache_file))
    data = cache.load()

    csi_original = data["csi"]
    mean_position = np.mean(data["pos"], axis=0)

    print()
    print(f"CSI original shape: {csi_original.shape}")
    print(f"Mean target position: {mean_position}")

    baseline_results = run_music_all_arrays(
        csi=csi_original,
        mean_position=mean_position,
        frequency_hz=frequency_hz,
        n_sources=n_sources,
    )

    print_results(
        title="PURE MUSIC - LUIS COVARIANCE 4x4",
        results=baseline_results,
    )

    print()
    print("=" * 70)
    print("APPLYING LUIS PCA PREPROCESSING")
    print("=" * 70)
    print("For each snapshot and each array:")
    print("  (2, 4, 53) -> transpose -> (53, 2, 4) -> reshape -> (53, 8)")
    print("  PCA(real) and PCA(imag) independently")
    print("  reconstruct and reshape back to (2, 4, 53)")

    csi_pca, pca_stats = luis_pca_per_snapshot_array(
        csi=csi_original,
        explained_variance=pca_variance,
    )

    print()
    print("PCA statistics:")
    for key, value in pca_stats.items():
        print(f"  {key}: {value}")

    pca_results = run_music_all_arrays(
        csi=csi_pca,
        mean_position=mean_position,
        frequency_hz=frequency_hz,
        n_sources=n_sources,
    )

    print_results(
        title="PCA REAL/IMAG PER SNAPSHOT/ARRAY + MUSIC",
        results=pca_results,
    )

    summary = {
        "experiment": "luis_sequence_adapted_pca_per_snapshot_array",
        "scenario": "human_helmet_standing_center_1",
        "crap_enabled": False,
        "frequency_hz": float(frequency_hz),
        "n_sources": int(n_sources),
        "pca_variance": float(pca_variance),
        "mean_target_position": mean_position.tolist(),
        "pca_stats": pca_stats,
        "baseline_results": baseline_results,
        "pca_results": pca_results,
    }

    save_json_metrics(summary, str(output_file))

    print()
    print(f"Metrics saved to: {output_file}")


if __name__ == "__main__":
    main()