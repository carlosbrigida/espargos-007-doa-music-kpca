from pathlib import Path
import sys
import json

import numpy as np
from sklearn.decomposition import PCA, KernelPCA

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

        n_snapshots = csi.shape[0]

        for start in range(0, n_snapshots - window_size + 1, stride):
            end = start + window_size
            csi_window = csi[start:end]

            R_all = covariance_per_array_luis(csi_window)

            for array_index in range(R_all.shape[0]):
                all_r.append(R_all[array_index])

    return np.asarray(all_r, dtype=np.complex64)


def estimate_music(
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
        "array_index": int(array_index),
        "array_name": array.name,
        "true_aoa_deg": float(true_angle),
        "estimated_aoa_deg": float(estimated_angle),
        "angular_error_deg": float(error),
        "eigenvalues": eigvals.real.tolist(),
    }


def evaluate_scenario(
    scenario_key: str,
    R_collection: np.ndarray,
    method_name: str,
    frequency_hz: float,
    n_sources: int,
) -> list[dict]:
    data = load_scenario(scenario_key)
    mean_position = np.mean(data["pos"], axis=0)

    results = []

    for array_index, R in enumerate(R_collection):
        result = estimate_music(
            R=R,
            array_index=array_index,
            mean_position=mean_position,
            frequency_hz=frequency_hz,
            n_sources=n_sources,
        )

        result["method"] = method_name
        result["scenario_key"] = scenario_key
        result["scenario_label"] = SCENARIOS[scenario_key]["label"]

        results.append(result)

    return results


def summarize(results: list[dict]) -> dict:
    errors = np.asarray(
        [item["angular_error_deg"] for item in results],
        dtype=float,
    )

    return {
        "mean_error_deg": float(np.mean(errors)),
        "median_error_deg": float(np.median(errors)),
        "min_error_deg": float(np.min(errors)),
        "max_error_deg": float(np.max(errors)),
    }


def print_results(title: str, results: list[dict]) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    for item in results:
        print(
            f"{item['scenario_label']} | "
            f"Array {item['array_index']} ({item['array_name']}): "
            f"true={item['true_aoa_deg']:.2f}° | "
            f"est={item['estimated_aoa_deg']:.2f}° | "
            f"err={item['angular_error_deg']:.2f}°"
        )

    summary = summarize(results)

    print()
    print(f"Mean error:   {summary['mean_error_deg']:.2f}°")
    print(f"Median error: {summary['median_error_deg']:.2f}°")


def train_apply_pca(
    R_train: np.ndarray,
    R_test: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, dict]:
    X_train = r_to_features(R_train)
    X_test = r_to_features(R_test)

    pca = PCA(n_components=n_components)
    pca.fit(X_train)

    X_rec = pca.inverse_transform(pca.transform(X_test))

    return features_to_r(X_rec), {
        "n_components": int(n_components),
        "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
    }


def train_apply_kpca(
    R_train: np.ndarray,
    R_test: np.ndarray,
    n_components: int,
    gamma: float,
) -> tuple[np.ndarray, dict]:
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

    return features_to_r(X_rec), {
        "n_components": int(n_components),
        "kernel": "rbf",
        "gamma": float(gamma),
    }


def main() -> None:
    frequency_hz = 2.472e9
    n_sources = 2

    window_size = 100
    stride = 50

    pca_components = [2, 4, 8]
    kpca_components = [2, 4, 8]
    kpca_gammas = [0.001, 0.01, 0.03125, 0.1]

    print()
    print("=" * 80)
    print("WINDOW-TRAINED PCA/KPCA ON R + MUSIC")
    print("=" * 80)
    print("Validation: leave-one-scenario-out")
    print("CRAP: disabled")
    print(f"Window size: {window_size}")
    print(f"Stride: {stride}")

    all_results = {
        "experiment": "window_trained_r_space_pca_kpca_music_loso_without_crap",
        "crap_enabled": False,
        "frequency_hz": float(frequency_hz),
        "n_sources": int(n_sources),
        "window_size": int(window_size),
        "stride": int(stride),
        "scenarios": {},
    }

    global_baseline = []
    global_pca_entries = []
    global_kpca_entries = []

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
        R_test = covariance_per_array_luis(test_data["csi"])

        print(f"Training R shape: {R_train.shape}")
        print(f"Test R shape: {R_test.shape}")

        baseline_results = evaluate_scenario(
            scenario_key=test_scenario,
            R_collection=R_test,
            method_name="pure_music",
            frequency_hz=frequency_hz,
            n_sources=n_sources,
        )

        print_results("PURE MUSIC", baseline_results)
        global_baseline.extend(baseline_results)

        scenario_output = {
            "train_scenarios": train_scenarios,
            "training_r_shape": list(R_train.shape),
            "test_r_shape": list(R_test.shape),
            "baseline": {
                "results": baseline_results,
                "summary": summarize(baseline_results),
            },
            "pca": [],
            "kpca": [],
        }

        for n_comp in pca_components:
            try:
                R_pca, pca_stats = train_apply_pca(
                    R_train=R_train,
                    R_test=R_test,
                    n_components=n_comp,
                )

                pca_results = evaluate_scenario(
                    scenario_key=test_scenario,
                    R_collection=R_pca,
                    method_name=f"pca_R_nc{n_comp}",
                    frequency_hz=frequency_hz,
                    n_sources=n_sources,
                )

                print_results(
                    f"PCA ON R | n_components={n_comp}",
                    pca_results,
                )

                entry = {
                    "stats": pca_stats,
                    "results": pca_results,
                    "summary": summarize(pca_results),
                }

                scenario_output["pca"].append(entry)
                global_pca_entries.append(entry)

            except Exception as exc:
                print(f"PCA failed for n_components={n_comp}: {exc}")

        for n_comp in kpca_components:
            for gamma in kpca_gammas:
                try:
                    R_kpca, kpca_stats = train_apply_kpca(
                        R_train=R_train,
                        R_test=R_test,
                        n_components=n_comp,
                        gamma=gamma,
                    )

                    kpca_results = evaluate_scenario(
                        scenario_key=test_scenario,
                        R_collection=R_kpca,
                        method_name=f"kpca_R_nc{n_comp}_g{gamma}",
                        frequency_hz=frequency_hz,
                        n_sources=n_sources,
                    )

                    print_results(
                        f"KPCA ON R | n_components={n_comp} | gamma={gamma}",
                        kpca_results,
                    )

                    entry = {
                        "stats": kpca_stats,
                        "results": kpca_results,
                        "summary": summarize(kpca_results),
                    }

                    scenario_output["kpca"].append(entry)
                    global_kpca_entries.append(entry)

                except Exception as exc:
                    print(
                        f"KPCA failed for n_components={n_comp}, "
                        f"gamma={gamma}: {exc}"
                    )

        all_results["scenarios"][test_scenario] = scenario_output

    best_pca = min(
        global_pca_entries,
        key=lambda item: item["summary"]["mean_error_deg"],
    )

    best_kpca = min(
        global_kpca_entries,
        key=lambda item: item["summary"]["mean_error_deg"],
    )

    all_results["global_summary"] = {
        "pure_music": summarize(global_baseline),
        "best_pca": {
            "stats": best_pca["stats"],
            "summary": best_pca["summary"],
        },
        "best_kpca": {
            "stats": best_kpca["stats"],
            "summary": best_kpca["summary"],
        },
    }

    output_file = (
        PROJECT_DIR
        / "outputs"
        / "metrics"
        / "window_trained_r_space_pca_kpca_music_loso_without_crap.json"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=4)

    print()
    print("=" * 80)
    print("FINAL GLOBAL SUMMARY")
    print("=" * 80)
    print(
        "Pure MUSIC mean error: "
        f"{all_results['global_summary']['pure_music']['mean_error_deg']:.2f}°"
    )
    print(
        "Best PCA mean error:  "
        f"{best_pca['summary']['mean_error_deg']:.2f}° | "
        f"n_components={best_pca['stats']['n_components']}"
    )
    print(
        "Best KPCA mean error: "
        f"{best_kpca['summary']['mean_error_deg']:.2f}° | "
        f"n_components={best_kpca['stats']['n_components']} | "
        f"gamma={best_kpca['stats']['gamma']}"
    )

    print()
    print(f"Metrics saved to: {output_file}")


if __name__ == "__main__":
    main()