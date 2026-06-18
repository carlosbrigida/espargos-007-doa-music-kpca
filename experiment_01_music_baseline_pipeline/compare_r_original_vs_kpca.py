from pathlib import Path
import sys
import json

import numpy as np
from sklearn.decomposition import KernelPCA

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.array_geometry import steering_vector_ula, wavelength_from_frequency
from src.cache_dataset import DatasetCache
from src.espargos_geometry import RECEIVER_ARRAYS, true_azimuth_deg
from src.metrics import angular_error_deg
from src.music import MusicEstimator


SCENARIOS = {
    "standing_center": {
        "label": "Standing Center",
        "cache": "data/cache/espargos_007_human_helmet_standing_center_1.npz",
    },
    "meanders_nw_se": {
        "label": "Meanders NW-SE",
        "cache": "data/cache/espargos_007_human_helmet_meanders_nw_se_1.npz",
    },
    "meanders_sw_ne": {
        "label": "Meanders SW-NE",
        "cache": "data/cache/espargos_007_human_helmet_meanders_sw_ne_1.npz",
    },
}


def covariance_per_array_luis(csi: np.ndarray) -> np.ndarray:
    return np.einsum(
        "dbrms,dbrns->bmn",
        csi,
        np.conj(csi),
    ) / csi.shape[0]


def r_to_features(R_collection: np.ndarray) -> np.ndarray:
    n_samples, n_antennas, _ = R_collection.shape
    R_flat = R_collection.reshape(n_samples, n_antennas * n_antennas)
    return np.concatenate([R_flat.real, R_flat.imag], axis=1)


def features_to_r(X: np.ndarray, n_antennas: int = 4) -> np.ndarray:
    n_flat = n_antennas * n_antennas

    real_part = X[:, :n_flat]
    imag_part = X[:, n_flat:]

    R_flat = real_part + 1j * imag_part
    R = R_flat.reshape(X.shape[0], n_antennas, n_antennas)
    R = 0.5 * (R + np.conj(np.transpose(R, (0, 2, 1))))

    return R.astype(np.complex64)


def load_scenario(scenario_key: str) -> dict:
    cache_file = PROJECT_DIR / SCENARIOS[scenario_key]["cache"]
    return DatasetCache(str(cache_file)).load()


def collect_window_covariances(
    scenario_keys: list[str],
    window_size: int,
    stride: int,
) -> np.ndarray:
    all_r = []

    for scenario_key in scenario_keys:
        data = load_scenario(scenario_key)
        csi = data["csi"]

        for start in range(0, csi.shape[0] - window_size + 1, stride):
            csi_window = csi[start:start + window_size]
            R_all = covariance_per_array_luis(csi_window)

            for array_index in range(R_all.shape[0]):
                all_r.append(R_all[array_index])

    return np.asarray(all_r, dtype=np.complex64)


def train_apply_kpca(
    R_train: np.ndarray,
    R_test: np.ndarray,
    n_components: int,
    gamma: float,
) -> np.ndarray:
    X_train = r_to_features(R_train)
    X_test = r_to_features(R_test)

    kpca = KernelPCA(
        n_components=n_components,
        kernel="rbf",
        gamma=gamma,
        fit_inverse_transform=True,
        alpha=1e-3,
    )

    kpca.fit(X_train)

    X_rec = kpca.inverse_transform(kpca.transform(X_test))

    return features_to_r(X_rec)


def music_spectrum(
    R: np.ndarray,
    array_index: int,
    mean_position: np.ndarray,
    frequency_hz: float,
    n_sources: int,
) -> dict:
    array = RECEIVER_ARRAYS[array_index]
    true_angle = true_azimuth_deg(array, mean_position)

    music = MusicEstimator(n_sources=n_sources)
    eigvals, eigvecs = music.eigendecomposition(R)
    En = music.noise_subspace(eigvecs)

    wavelength = wavelength_from_frequency(frequency_hz)
    antenna_spacing = wavelength / 2.0

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
        noise_subspace=En,
        steering_vectors=steering_vectors,
    )

    estimated_angle = float(angles_deg[np.argmax(spectrum)])
    error = angular_error_deg(estimated_angle, true_angle)

    return {
        "true_angle_deg": float(true_angle),
        "estimated_angle_deg": float(estimated_angle),
        "error_deg": float(error),
        "angles_deg": angles_deg,
        "spectrum": spectrum,
        "eigenvalues": eigvals.real,
    }


def matrix_distance_metrics(
    R_original: np.ndarray,
    R_kpca: np.ndarray,
) -> dict:
    diff = R_original - R_kpca

    norm_original = np.linalg.norm(R_original)
    norm_kpca = np.linalg.norm(R_kpca)
    norm_diff = np.linalg.norm(diff)

    relative_diff = norm_diff / norm_original if norm_original > 0 else np.nan

    eig_original = np.linalg.eigvalsh(R_original)[::-1].real
    eig_kpca = np.linalg.eigvalsh(R_kpca)[::-1].real

    return {
        "norm_original": float(norm_original),
        "norm_kpca": float(norm_kpca),
        "norm_difference": float(norm_diff),
        "relative_difference": float(relative_diff),
        "eigenvalues_original": eig_original.tolist(),
        "eigenvalues_kpca": eig_kpca.tolist(),
    }


def main() -> None:
    frequency_hz = 2.472e9
    n_sources = 2

    window_size = 100
    stride = 50

    best_n_components = 2
    best_gamma = 0.01

    print()
    print("=" * 80)
    print("COMPARE R ORIGINAL VS R KPCA")
    print("=" * 80)
    print("Goal: verify whether KPCA preserves angular information")
    print("or collapses R matrices toward a common average solution.")
    print()
    print(f"KPCA: n_components={best_n_components}, gamma={best_gamma}")
    print(f"Window size: {window_size}")
    print(f"Stride: {stride}")

    all_reports = {}

    for test_scenario in SCENARIOS:
        train_scenarios = [
            key for key in SCENARIOS if key != test_scenario
        ]

        print()
        print("#" * 80)
        print(f"TEST SCENARIO: {SCENARIOS[test_scenario]['label']}")
        print(f"TRAIN SCENARIOS: {[SCENARIOS[k]['label'] for k in train_scenarios]}")
        print("#" * 80)

        R_train = collect_window_covariances(
            scenario_keys=train_scenarios,
            window_size=window_size,
            stride=stride,
        )

        test_data = load_scenario(test_scenario)
        csi_test = test_data["csi"]
        mean_position = np.mean(test_data["pos"], axis=0)

        R_original_all = covariance_per_array_luis(csi_test)

        R_kpca_all = train_apply_kpca(
            R_train=R_train,
            R_test=R_original_all,
            n_components=best_n_components,
            gamma=best_gamma,
        )

        scenario_report = {
            "train_scenarios": train_scenarios,
            "training_r_shape": list(R_train.shape),
            "arrays": [],
        }

        for array_index in range(4):
            R_original = R_original_all[array_index]
            R_kpca = R_kpca_all[array_index]

            original_music = music_spectrum(
                R=R_original,
                array_index=array_index,
                mean_position=mean_position,
                frequency_hz=frequency_hz,
                n_sources=n_sources,
            )

            kpca_music = music_spectrum(
                R=R_kpca,
                array_index=array_index,
                mean_position=mean_position,
                frequency_hz=frequency_hz,
                n_sources=n_sources,
            )

            metrics = matrix_distance_metrics(
                R_original=R_original,
                R_kpca=R_kpca,
            )

            report = {
                "array_index": int(array_index),
                "array_name": RECEIVER_ARRAYS[array_index].name,
                "matrix_metrics": metrics,
                "music_original": {
                    "true_angle_deg": original_music["true_angle_deg"],
                    "estimated_angle_deg": original_music["estimated_angle_deg"],
                    "error_deg": original_music["error_deg"],
                    "eigenvalues": original_music["eigenvalues"].tolist(),
                },
                "music_kpca": {
                    "true_angle_deg": kpca_music["true_angle_deg"],
                    "estimated_angle_deg": kpca_music["estimated_angle_deg"],
                    "error_deg": kpca_music["error_deg"],
                    "eigenvalues": kpca_music["eigenvalues"].tolist(),
                },
            }

            scenario_report["arrays"].append(report)

            print()
            print(f"Array {array_index} - {RECEIVER_ARRAYS[array_index].name}")
            print(
                f"  Original MUSIC: "
                f"est={original_music['estimated_angle_deg']:.2f}° | "
                f"err={original_music['error_deg']:.2f}°"
            )
            print(
                f"  KPCA MUSIC:     "
                f"est={kpca_music['estimated_angle_deg']:.2f}° | "
                f"err={kpca_music['error_deg']:.2f}°"
            )
            print(
                f"  ||R - R_kpca|| / ||R|| = "
                f"{metrics['relative_difference']:.4f}"
            )
            print(
                f"  eig original: "
                f"{np.round(metrics['eigenvalues_original'], 2)}"
            )
            print(
                f"  eig kpca:     "
                f"{np.round(metrics['eigenvalues_kpca'], 2)}"
            )

        all_reports[test_scenario] = scenario_report

    output_file = (
        PROJECT_DIR
        / "outputs"
        / "metrics"
        / "compare_r_original_vs_kpca.json"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            {
                "experiment": "compare_r_original_vs_kpca",
                "kpca": {
                    "n_components": best_n_components,
                    "gamma": best_gamma,
                    "kernel": "rbf",
                },
                "window_size": window_size,
                "stride": stride,
                "reports": all_reports,
            },
            file,
            indent=4,
        )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Metrics saved to: {output_file}")


if __name__ == "__main__":
    main()