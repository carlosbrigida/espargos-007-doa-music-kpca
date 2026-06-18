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


def features_to_r(
    X_features: np.ndarray,
    n_antennas: int = 4,
) -> np.ndarray:
    n_flat = n_antennas * n_antennas

    real_part = X_features[:, :n_flat]
    imag_part = X_features[:, n_flat:]

    R_flat = real_part + 1j * imag_part
    R = R_flat.reshape(X_features.shape[0], n_antennas, n_antennas)

    R = 0.5 * (R + np.conj(np.transpose(R, (0, 2, 1))))

    return R.astype(np.complex64)


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

    spectrum = music.pseudo_spectrum(En, steering_vectors)

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


def load_all_r_matrices() -> tuple[list[dict], np.ndarray]:
    records = []
    r_list = []

    for scenario_key, cfg in SCENARIOS.items():
        cache = DatasetCache(str(PROJECT_DIR / cfg["cache"]))
        data = cache.load()

        csi = data["csi"]
        mean_position = np.mean(data["pos"], axis=0)

        R_all = covariance_per_array_luis(csi)

        for array_index in range(R_all.shape[0]):
            records.append(
                {
                    "scenario_key": scenario_key,
                    "scenario_label": cfg["label"],
                    "array_index": int(array_index),
                    "array_name": RECEIVER_ARRAYS[array_index].name,
                    "mean_position": mean_position,
                }
            )
            r_list.append(R_all[array_index])

    return records, np.asarray(r_list, dtype=np.complex64)


def evaluate_r_collection(
    records: list[dict],
    R_collection: np.ndarray,
    method_name: str,
    frequency_hz: float,
    n_sources: int,
) -> list[dict]:
    results = []

    for idx, R in enumerate(R_collection):
        record = records[idx]

        music_result = estimate_music(
            R=R,
            array_index=record["array_index"],
            mean_position=record["mean_position"],
            frequency_hz=frequency_hz,
            n_sources=n_sources,
        )

        results.append(
            {
                "method": method_name,
                "scenario_key": record["scenario_key"],
                "scenario_label": record["scenario_label"],
                "array_index": record["array_index"],
                "array_name": record["array_name"],
                **music_result,
            }
        )

    return results


def apply_pca_to_r(
    R_collection: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, dict]:
    X = r_to_features(R_collection)

    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)
    X_rec = pca.inverse_transform(Z)

    R_rec = features_to_r(X_rec)

    stats = {
        "n_components": int(n_components),
        "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
    }

    return R_rec, stats


def apply_kpca_to_r(
    R_collection: np.ndarray,
    n_components: int,
    gamma: float,
) -> tuple[np.ndarray, dict]:
    X = r_to_features(R_collection)

    kpca = KernelPCA(
        n_components=n_components,
        kernel="rbf",
        gamma=gamma,
        fit_inverse_transform=True,
        alpha=1e-3,
    )

    Z = kpca.fit_transform(X)
    X_rec = kpca.inverse_transform(Z)

    R_rec = features_to_r(X_rec)

    stats = {
        "n_components": int(n_components),
        "kernel": "rbf",
        "gamma": float(gamma),
    }

    return R_rec, stats


def summarize(results: list[dict]) -> dict:
    errors = np.asarray([item["angular_error_deg"] for item in results])

    return {
        "mean_error_deg": float(np.mean(errors)),
        "median_error_deg": float(np.median(errors)),
        "min_error_deg": float(np.min(errors)),
        "max_error_deg": float(np.max(errors)),
    }


def print_method_results(title: str, results: list[dict]) -> None:
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

    pca_components = [2, 4, 8]
    kpca_components = [2, 4, 8]
    kpca_gammas = [0.001, 0.01, 0.03125, 0.1]

    print()
    print("=" * 80)
    print("R-SPACE PCA/KPCA + MUSIC EXPERIMENT")
    print("=" * 80)
    print("CRAP: disabled")
    print("Representation:")
    print("  R shape: 4x4 complex")
    print("  flattened: 16 complex")
    print("  real+imag: 32 real features")
    print()

    records, R_collection = load_all_r_matrices()

    print(f"Total R matrices: {R_collection.shape[0]}")
    print(f"R collection shape: {R_collection.shape}")

    baseline_results = evaluate_r_collection(
        records=records,
        R_collection=R_collection,
        method_name="pure_music",
        frequency_hz=frequency_hz,
        n_sources=n_sources,
    )

    print_method_results("PURE MUSIC", baseline_results)

    all_outputs = {
        "experiment": "r_space_pca_kpca_music_without_crap",
        "frequency_hz": float(frequency_hz),
        "n_sources": int(n_sources),
        "crap_enabled": False,
        "baseline": {
            "results": baseline_results,
            "summary": summarize(baseline_results),
        },
        "pca": [],
        "kpca": [],
    }

    best_pca = None
    best_kpca = None

    for n_comp in pca_components:
        R_pca, pca_stats = apply_pca_to_r(
            R_collection=R_collection,
            n_components=n_comp,
        )

        pca_results = evaluate_r_collection(
            records=records,
            R_collection=R_pca,
            method_name=f"pca_R_nc{n_comp}",
            frequency_hz=frequency_hz,
            n_sources=n_sources,
        )

        pca_summary = summarize(pca_results)

        print_method_results(
            f"PCA ON R | n_components={n_comp} | "
            f"variance={pca_stats['explained_variance']:.4f}",
            pca_results,
        )

        entry = {
            "stats": pca_stats,
            "results": pca_results,
            "summary": pca_summary,
        }

        all_outputs["pca"].append(entry)

        if best_pca is None or pca_summary["mean_error_deg"] < best_pca["summary"]["mean_error_deg"]:
            best_pca = entry

    for n_comp in kpca_components:
        for gamma in kpca_gammas:
            try:
                R_kpca, kpca_stats = apply_kpca_to_r(
                    R_collection=R_collection,
                    n_components=n_comp,
                    gamma=gamma,
                )

                kpca_results = evaluate_r_collection(
                    records=records,
                    R_collection=R_kpca,
                    method_name=f"kpca_R_nc{n_comp}_g{gamma}",
                    frequency_hz=frequency_hz,
                    n_sources=n_sources,
                )

                kpca_summary = summarize(kpca_results)

                print_method_results(
                    f"KPCA ON R | n_components={n_comp} | gamma={gamma}",
                    kpca_results,
                )

                entry = {
                    "stats": kpca_stats,
                    "results": kpca_results,
                    "summary": kpca_summary,
                }

                all_outputs["kpca"].append(entry)

                if best_kpca is None or kpca_summary["mean_error_deg"] < best_kpca["summary"]["mean_error_deg"]:
                    best_kpca = entry

            except Exception as exc:
                print()
                print(f"KPCA failed for n_components={n_comp}, gamma={gamma}")
                print(f"Error: {exc}")

    all_outputs["best_pca"] = best_pca
    all_outputs["best_kpca"] = best_kpca

    output_file = (
        PROJECT_DIR
        / "outputs"
        / "metrics"
        / "r_space_pca_kpca_music_without_crap.json"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(all_outputs, file, indent=4)

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Pure MUSIC mean error: {all_outputs['baseline']['summary']['mean_error_deg']:.2f}°")

    if best_pca is not None:
        print(
            "Best PCA mean error:  "
            f"{best_pca['summary']['mean_error_deg']:.2f}° | "
            f"n_components={best_pca['stats']['n_components']}"
        )

    if best_kpca is not None:
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