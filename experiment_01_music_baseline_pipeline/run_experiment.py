import sys

import numpy as np

from src.array_geometry import steering_vector_azimuth_3d
from src.cache_dataset import DatasetCache
from src.data_loader import EspargosTFRecordLoader
from src.espargos_geometry import (
    RECEIVER_ARRAYS,
    antenna_positions,
    array_normal,
    true_azimuth_deg,
)
from src.metrics import angular_error_deg, save_json_metrics
from src.music import MusicEstimator
from src.preprocessing import covariance_per_array_8_antennas
from src.visualization import (
    plot_method_heatmap,
    plot_three_method_mean_comparison,
)


SCENARIOS = {
    "standing_center": {
        "name": "human_helmet_standing_center_1",
        "label": "Standing Center",
        "tfrecord": (
            r"C:\Users\Carlos\Documents\Doutorado\Espargos 007\Dados"
            r"\espargos-0007-human-helmet-standing-center-1.tfrecords"
        ),
        "cache": "data/cache/espargos_007_human_helmet_standing_center_1.npz",
    },
    "meanders_nw_se": {
        "name": "human_helmet_meanders_nw_se_1",
        "label": "Meanders NW-SE",
        "tfrecord": (
            r"C:\Users\Carlos\Documents\Doutorado\Espargos 007\Dados"
            r"\espargos-0007-human-helmet-meanders-nw-se-1.tfrecords"
        ),
        "cache": "data/cache/espargos_007_human_helmet_meanders_nw_se_1.npz",
    },
    "meanders_sw_ne": {
        "name": "human_helmet_meanders_sw_ne_1",
        "label": "Meanders SW-NE",
        "tfrecord": (
            r"C:\Users\Carlos\Documents\Doutorado\Espargos 007\Dados"
            r"\espargos-0007-human-helmet-meanders-sw-ne-1.tfrecords"
        ),
        "cache": "data/cache/espargos_007_human_helmet_meanders_sw_ne_1.npz",
    },
}


def load_or_create_cache(cache_file: str, tfrecord_file: str) -> dict:
    cache = DatasetCache(cache_file)

    try:
        return cache.load()
    except FileNotFoundError:
        loader = EspargosTFRecordLoader([tfrecord_file])
        data = loader.load()
        cache.save(data)
        return data


def estimate_music_8_antennas(
    R: np.ndarray,
    array_index: int,
    mean_position: np.ndarray,
    frequency_hz: float,
    n_sources: int,
) -> dict:
    array = RECEIVER_ARRAYS[array_index]

    true_angle_deg = true_azimuth_deg(array, mean_position)

    positions = antenna_positions(array)
    normal = array_normal(array)

    music = MusicEstimator(n_sources=n_sources)

    eigvals, eigvecs = music.eigendecomposition(R)
    noise_subspace = music.noise_subspace(eigvecs)

    angles_deg = np.linspace(-90.0, 90.0, 721)

    steering_vectors = np.asarray(
        [
            steering_vector_azimuth_3d(
                angle_deg=angle,
                antenna_positions=positions,
                array_center=array.center,
                array_right=array.right,
                array_normal=normal,
                frequency_hz=frequency_hz,
            )
            for angle in angles_deg
        ]
    )

    spectrum = music.pseudo_spectrum(noise_subspace, steering_vectors)

    estimated_angle_deg = float(angles_deg[np.argmax(spectrum)])
    error_deg = angular_error_deg(estimated_angle_deg, true_angle_deg)

    return {
        "array_index": int(array_index),
        "array_name": array.name,
        "true_aoa_deg": float(true_angle_deg),
        "estimated_aoa_deg": float(estimated_angle_deg),
        "angular_error_deg": float(error_deg),
        "eigenvalues": eigvals.real.tolist(),
    }


def estimate_internal_pca_music_8_antennas(
    R: np.ndarray,
    array_index: int,
    mean_position: np.ndarray,
    frequency_hz: float,
    n_sources: int,
    variance_threshold: float,
) -> dict:
    music = MusicEstimator(n_sources=n_sources)

    R_pca, n_components, preserved_variance = (
        music.reconstruct_covariance_by_variance(
            R=R,
            variance_threshold=variance_threshold,
        )
    )

    result = estimate_music_8_antennas(
        R=R_pca,
        array_index=array_index,
        mean_position=mean_position,
        frequency_hz=frequency_hz,
        n_sources=n_sources,
    )

    result["internal_pca_variance_threshold"] = float(variance_threshold)
    result["internal_pca_components"] = int(n_components)
    result["internal_pca_preserved_variance"] = float(preserved_variance)

    return result


def run_scenario(
    scenario_key: str,
    scenario_cfg: dict,
    frequency_hz: float,
    n_sources: int,
    pca_variances: list[float],
) -> dict:
    data = load_or_create_cache(
        cache_file=scenario_cfg["cache"],
        tfrecord_file=scenario_cfg["tfrecord"],
    )

    csi = data["csi"]
    mean_position = np.mean(data["pos"], axis=0)

    R_all = covariance_per_array_8_antennas(csi)

    print()
    print("===================================")
    print(f"Scenario: {scenario_cfg['label']}")
    print(f"CSI shape: {csi.shape}")
    print(f"R_all shape: {R_all.shape}")

    pure_results = []
    internal_pca_results = []

    for array_index in range(R_all.shape[0]):
        R = R_all[array_index]

        pure_result = estimate_music_8_antennas(
            R=R,
            array_index=array_index,
            mean_position=mean_position,
            frequency_hz=frequency_hz,
            n_sources=n_sources,
        )

        pure_results.append(pure_result)

        print(
            f"Pure MUSIC 8-ant | Array {array_index} "
            f"({pure_result['array_name']}): "
            f"Err={pure_result['angular_error_deg']:.2f}° "
            f"Est={pure_result['estimated_aoa_deg']:.2f}° "
            f"True={pure_result['true_aoa_deg']:.2f}°"
        )

        array_pca_results = []

        for variance in pca_variances:
            pca_result = estimate_internal_pca_music_8_antennas(
                R=R,
                array_index=array_index,
                mean_position=mean_position,
                frequency_hz=frequency_hz,
                n_sources=n_sources,
                variance_threshold=variance,
            )

            array_pca_results.append(pca_result)

            print(
                f"  Internal PCA {variance:.2f} | "
                f"Comp={pca_result['internal_pca_components']} | "
                f"Preserved={pca_result['internal_pca_preserved_variance']:.4f} | "
                f"Err={pca_result['angular_error_deg']:.2f}°"
            )

        internal_pca_results.append(
            {
                "array_index": int(array_index),
                "array_name": pure_result["array_name"],
                "results": array_pca_results,
            }
        )

    pure_errors = np.asarray(
        [item["angular_error_deg"] for item in pure_results],
        dtype=float,
    )

    best_internal_errors = []
    best_internal_metadata = []

    for array_group in internal_pca_results:
        best = min(
            array_group["results"],
            key=lambda item: item["angular_error_deg"],
        )

        best_internal_errors.append(best["angular_error_deg"])

        best_internal_metadata.append(
            {
                "array_index": int(array_group["array_index"]),
                "array_name": array_group["array_name"],
                "best_error_deg": float(best["angular_error_deg"]),
                "best_variance_threshold": float(
                    best["internal_pca_variance_threshold"]
                ),
                "best_components": int(best["internal_pca_components"]),
                "best_preserved_variance": float(
                    best["internal_pca_preserved_variance"]
                ),
            }
        )

    return {
        "scenario_key": scenario_key,
        "scenario_name": scenario_cfg["name"],
        "scenario_label": scenario_cfg["label"],
        "csi_shape": list(csi.shape),
        "covariance_shape": list(R_all.shape),
        "mean_position": mean_position.tolist(),
        "pure_music_results": pure_results,
        "internal_pca_results": internal_pca_results,
        "pure_music_errors_by_array": pure_errors.tolist(),
        "best_internal_pca_errors_by_array": best_internal_errors,
        "best_internal_pca_metadata": best_internal_metadata,
    }


def main() -> None:
    frequency_hz = 2.472e9
    n_sources = 2
    pca_variances = [0.90, 0.95, 0.99]

    scenario_arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if scenario_arg == "all":
        selected_scenarios = SCENARIOS
    else:
        selected_scenarios = {scenario_arg: SCENARIOS[scenario_arg]}

    print("Internal PCA inside MUSIC — 8 antenna covariance")
    print("CRAP: disabled")
    print(f"n_sources: {n_sources}")

    all_results = []

    for scenario_key, scenario_cfg in selected_scenarios.items():
        result = run_scenario(
            scenario_key=scenario_key,
            scenario_cfg=scenario_cfg,
            frequency_hz=frequency_hz,
            n_sources=n_sources,
            pca_variances=pca_variances,
        )
        all_results.append(result)

    scenario_labels = [item["scenario_label"] for item in all_results]
    array_labels = [
        f"A{idx}: {RECEIVER_ARRAYS[idx].name.replace('espargos', '')}"
        for idx in range(4)
    ]

    pure_matrix = np.asarray(
        [item["pure_music_errors_by_array"] for item in all_results],
        dtype=float,
    )

    internal_matrix = np.asarray(
        [item["best_internal_pca_errors_by_array"] for item in all_results],
        dtype=float,
    )

    output_metrics = (
        "outputs/metrics/"
        "internal_pca_music_8_antennas_without_crap.json"
    )

    save_json_metrics(
        {
            "experiment": "internal_pca_music_8_antennas_without_crap",
            "crap_enabled": False,
            "frequency_hz": float(frequency_hz),
            "n_sources": int(n_sources),
            "pca_variances": pca_variances,
            "results": all_results,
            "pure_music_error_matrix": pure_matrix.tolist(),
            "best_internal_pca_error_matrix": internal_matrix.tolist(),
            "mean_errors": {
                "pure_music_8_antennas": float(np.mean(pure_matrix)),
                "internal_pca_music_8_antennas": float(np.mean(internal_matrix)),
            },
        },
        output_metrics,
    )

    plot_method_heatmap(
        values=pure_matrix,
        row_labels=scenario_labels,
        col_labels=array_labels,
        title="Pure MUSIC 8-Antenna Angular Error",
        output_file="outputs/figures/heatmap_pure_music_8ant.png",
    )

    plot_method_heatmap(
        values=internal_matrix,
        row_labels=scenario_labels,
        col_labels=array_labels,
        title="Internal PCA + MUSIC 8-Antenna Angular Error",
        output_file="outputs/figures/heatmap_internal_pca_music_8ant.png",
    )

    plot_three_method_mean_comparison(
        method_labels=[
            "Pure MUSIC 8-ant",
            "Internal PCA + MUSIC 8-ant",
        ],
        mean_errors=np.asarray(
            [
                np.mean(pure_matrix),
                np.mean(internal_matrix),
            ],
            dtype=float,
        ),
        output_file="outputs/figures/mean_error_internal_pca_8ant.png",
    )

    print()
    print("===================================")
    print("Mean error across scenarios and arrays")
    print("===================================")
    print(f"Pure MUSIC 8-ant:           {np.mean(pure_matrix):.2f}°")
    print(f"Internal PCA + MUSIC 8-ant: {np.mean(internal_matrix):.2f}°")
    print()
    print(f"Metrics saved to: {output_metrics}")


if __name__ == "__main__":
    main()