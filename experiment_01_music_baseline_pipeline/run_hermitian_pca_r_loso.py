from pathlib import Path
import sys
import json

import numpy as np
from sklearn.decomposition import PCA

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
    return np.einsum("dbrms,dbrns->bmn", csi, np.conj(csi)) / csi.shape[0]


def load_scenario(scenario_key: str) -> dict:
    cache_file = PROJECT_DIR / SCENARIOS[scenario_key]["cache"]
    return DatasetCache(str(cache_file)).load()


def r_to_hermitian_features(R_collection: np.ndarray) -> np.ndarray:
    features = []

    for R in R_collection:
        row = []

        for i in range(4):
            row.append(R[i, i].real)

        for i in range(4):
            for j in range(i + 1, 4):
                row.append(R[i, j].real)
                row.append(R[i, j].imag)

        features.append(row)

    return np.asarray(features, dtype=np.float64)


def hermitian_features_to_r(X: np.ndarray) -> np.ndarray:
    matrices = []

    for row in X:
        R = np.zeros((4, 4), dtype=np.complex64)

        idx = 0

        for i in range(4):
            R[i, i] = row[idx] + 0j
            idx += 1

        for i in range(4):
            for j in range(i + 1, 4):
                value = row[idx] + 1j * row[idx + 1]
                R[i, j] = value
                R[j, i] = np.conj(value)
                idx += 2

        matrices.append(R)

    return np.asarray(matrices, dtype=np.complex64)


def collect_window_covariances_by_array(
    scenario_keys: list[str],
    window_size: int,
    stride: int,
) -> dict[int, np.ndarray]:
    by_array = {array_index: [] for array_index in range(4)}

    for scenario_key in scenario_keys:
        data = load_scenario(scenario_key)
        csi = data["csi"]

        for start in range(0, csi.shape[0] - window_size + 1, stride):
            csi_window = csi[start:start + window_size]
            R_all = covariance_per_array_luis(csi_window)

            for array_index in range(4):
                by_array[array_index].append(R_all[array_index])

    return {
        array_index: np.asarray(values, dtype=np.complex64)
        for array_index, values in by_array.items()
    }


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
    noise_subspace = music.noise_subspace(eigvecs)

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


def summarize(results: list[dict]) -> dict:
    errors = np.asarray([item["angular_error_deg"] for item in results], dtype=float)

    return {
        "mean_error_deg": float(np.mean(errors)),
        "median_error_deg": float(np.median(errors)),
        "min_error_deg": float(np.min(errors)),
        "max_error_deg": float(np.max(errors)),
    }


def train_apply_hermitian_pca_per_array(
    R_train_by_array: dict[int, np.ndarray],
    R_test_all: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, dict]:
    reconstructed = []

    stats = {
        "n_components": int(n_components),
        "feature_type": "hermitian_independent_entries",
        "n_features": 16,
        "arrays": [],
    }

    for array_index in range(4):
        R_train = R_train_by_array[array_index]
        R_test = R_test_all[array_index:array_index + 1]

        X_train = r_to_hermitian_features(R_train)
        X_test = r_to_hermitian_features(R_test)

        pca = PCA(n_components=n_components)
        pca.fit(X_train)

        X_rec = pca.inverse_transform(pca.transform(X_test))
        R_rec = hermitian_features_to_r(X_rec)[0]

        reconstructed.append(R_rec)

        stats["arrays"].append(
            {
                "array_index": int(array_index),
                "array_name": RECEIVER_ARRAYS[array_index].name,
                "training_samples": int(R_train.shape[0]),
                "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
            }
        )

    return np.asarray(reconstructed, dtype=np.complex64), stats


def evaluate_scenario(
    scenario_key: str,
    R_all: np.ndarray,
    method_name: str,
    frequency_hz: float,
    n_sources: int,
) -> list[dict]:
    data = load_scenario(scenario_key)
    mean_position = np.mean(data["pos"], axis=0)

    results = []

    for array_index in range(4):
        result = estimate_music(
            R=R_all[array_index],
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


def main() -> None:
    frequency_hz = 2.472e9
    n_sources = 2

    window_size = 100
    stride = 50

    pca_components = [2, 4, 8, 12, 16]

    print()
    print("=" * 80)
    print("HERMITIAN PCA ON R + MUSIC - LOSO PER ARRAY")
    print("=" * 80)
    print("CRAP: disabled")
    print("Purpose: diagnostic experiment")
    print("R representation: 16 independent Hermitian features")
    print(f"Window size: {window_size}")
    print(f"Stride: {stride}")

    all_results = {
        "experiment": "hermitian_pca_r_loso_per_array_without_crap",
        "crap_enabled": False,
        "frequency_hz": float(frequency_hz),
        "n_sources": int(n_sources),
        "window_size": int(window_size),
        "stride": int(stride),
        "scenarios": {},
    }

    global_baseline = []
    global_pca_entries = []

    for test_scenario in SCENARIOS:
        train_scenarios = [key for key in SCENARIOS if key != test_scenario]

        print()
        print("#" * 80)
        print(f"TEST SCENARIO: {SCENARIOS[test_scenario]['label']}")
        print(f"TRAIN SCENARIOS: {[SCENARIOS[k]['label'] for k in train_scenarios]}")
        print("#" * 80)

        R_train_by_array = collect_window_covariances_by_array(
            scenario_keys=train_scenarios,
            window_size=window_size,
            stride=stride,
        )

        test_data = load_scenario(test_scenario)
        R_test_all = covariance_per_array_luis(test_data["csi"])

        baseline_results = evaluate_scenario(
            scenario_key=test_scenario,
            R_all=R_test_all,
            method_name="pure_music",
            frequency_hz=frequency_hz,
            n_sources=n_sources,
        )

        print_results("PURE MUSIC", baseline_results)
        global_baseline.extend(baseline_results)

        scenario_output = {
            "train_scenarios": train_scenarios,
            "baseline": {
                "results": baseline_results,
                "summary": summarize(baseline_results),
            },
            "hermitian_pca": [],
        }

        for n_comp in pca_components:
            R_pca_all, pca_stats = train_apply_hermitian_pca_per_array(
                R_train_by_array=R_train_by_array,
                R_test_all=R_test_all,
                n_components=n_comp,
            )

            pca_results = evaluate_scenario(
                scenario_key=test_scenario,
                R_all=R_pca_all,
                method_name=f"hermitian_pca_nc{n_comp}",
                frequency_hz=frequency_hz,
                n_sources=n_sources,
            )

            print_results(
                f"HERMITIAN PCA ON R | n_components={n_comp}",
                pca_results,
            )

            entry = {
                "stats": pca_stats,
                "results": pca_results,
                "summary": summarize(pca_results),
            }

            scenario_output["hermitian_pca"].append(entry)
            global_pca_entries.append(entry)

        all_results["scenarios"][test_scenario] = scenario_output

    best_pca = min(
        global_pca_entries,
        key=lambda item: item["summary"]["mean_error_deg"],
    )

    all_results["global_summary"] = {
        "pure_music": summarize(global_baseline),
        "best_hermitian_pca": {
            "stats": best_pca["stats"],
            "summary": best_pca["summary"],
        },
    }

    output_file = (
        PROJECT_DIR
        / "outputs"
        / "metrics"
        / "hermitian_pca_r_loso_per_array_without_crap.json"
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
        "Best Hermitian PCA mean error: "
        f"{best_pca['summary']['mean_error_deg']:.2f}° | "
        f"n_components={best_pca['stats']['n_components']}"
    )

    print()
    print(f"Metrics saved to: {output_file}")


if __name__ == "__main__":
    main()