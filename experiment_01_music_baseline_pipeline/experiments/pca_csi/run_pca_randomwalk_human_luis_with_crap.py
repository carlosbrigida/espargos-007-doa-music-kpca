from pathlib import Path
import sys
import json
import time

import numpy as np
from sklearn.decomposition import PCA

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.cache_dataset import DatasetCache
from src.espargos_geometry import RECEIVER_ARRAYS, true_azimuth_deg
from src.metrics import angular_error_deg
from src.run_uroot_music_luis_cache_with_crap import (
    remove_clutter,
    cluster_cache_data,
    covariance_per_array_luis_cluster,
    get_unitary_rootmusic_estimator,
)


CACHE_FILE = (
    PROJECT_DIR
    / "data"
    / "cache_luis"
    / "espargos_007_human_helmet_randomwalk_1.npz"
)

CLUTTER_FILE = (
    PROJECT_DIR
    / "clutter_channel_estimates_luis"
    / "espargos_007_human_helmet_randomwalk_1.npz.npy"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "metrics"
    / "pca_randomwalk_human_luis_with_crap.json"
)

CLUSTER_INTERVAL_SECONDS = 1.0
VARIANCES = [0.90, 0.95, 0.99]


def apply_crap_to_full_csi(csi, mac, clutter_acquisitions):
    csi_noclutter = np.empty_like(csi)
    unique_macs = np.unique(mac)

    for tx_index, current_mac in enumerate(unique_macs):
        tx_mask = mac == current_mac
        tx_csi = csi[tx_mask]

        tx_csi_noclutter = remove_clutter(
            tx_csi,
            clutter_acquisitions[tx_index],
        )

        csi_noclutter[tx_mask] = tx_csi_noclutter

    return csi_noclutter


def flatten_csi(csi):
    return csi.reshape(csi.shape[0], -1)


def unflatten_csi(x_flat, original_shape):
    return x_flat.reshape(original_shape)


def complex_pca_reconstruct(csi, variance_threshold):
    original_shape = csi.shape
    x = flatten_csi(csi)

    mean = np.mean(x, axis=0, keepdims=True)
    x_centered = x - mean

    u, singular_values, vh = np.linalg.svd(
        x_centered,
        full_matrices=False,
    )

    energy = singular_values ** 2
    cumulative_energy = np.cumsum(energy) / np.sum(energy)

    n_components = int(np.searchsorted(cumulative_energy, variance_threshold) + 1)

    x_reconstructed = (
        u[:, :n_components]
        @ np.diag(singular_values[:n_components])
        @ vh[:n_components, :]
    )

    x_reconstructed = x_reconstructed + mean
    preserved_energy = float(cumulative_energy[n_components - 1])

    return (
        unflatten_csi(x_reconstructed.astype(np.complex64), original_shape),
        n_components,
        preserved_energy,
    )


def real_imag_pca_reconstruct(csi, variance_threshold):
    original_shape = csi.shape
    x = flatten_csi(csi)

    x_realimag = np.concatenate([x.real, x.imag], axis=1)

    pca = PCA(n_components=variance_threshold, svd_solver="full")
    z = pca.fit_transform(x_realimag)
    x_reconstructed_realimag = pca.inverse_transform(z)

    n_features = x.shape[1]

    real_part = x_reconstructed_realimag[:, :n_features]
    imag_part = x_reconstructed_realimag[:, n_features:]

    x_reconstructed = real_part + 1j * imag_part

    n_components = int(pca.n_components_)
    preserved_energy = float(np.sum(pca.explained_variance_ratio_))

    return (
        unflatten_csi(x_reconstructed.astype(np.complex64), original_shape),
        n_components,
        preserved_energy,
    )


def evaluate_csi(csi, mac, pos, timestamps):
    clusters = cluster_cache_data(
        csi=csi,
        mac=mac,
        pos=pos,
        timestamps=timestamps,
        cluster_interval=CLUSTER_INTERVAL_SECONDS,
    )

    estimator = get_unitary_rootmusic_estimator(4)
    errors_by_array = {array_index: [] for array_index in range(4)}

    for cluster in clusters:
        covariance_all = covariance_per_array_luis_cluster(
            cluster["csi_by_transmitter"]
        )

        for array_index in range(4):
            array = RECEIVER_ARRAYS[array_index]
            true_angle_deg = true_azimuth_deg(array, cluster["mean_position"])

            electrical_angle, _ = estimator(covariance_all[array_index])

            if np.isnan(electrical_angle):
                continue

            estimated_angle_rad = np.arcsin(
                np.clip(electrical_angle / np.pi, -1.0, 1.0)
            )

            estimated_angle_deg = np.rad2deg(estimated_angle_rad)
            error = angular_error_deg(estimated_angle_deg, true_angle_deg)

            errors_by_array[array_index].append(error)

    mae_by_array = {
        array_index: float(np.mean(errors))
        for array_index, errors in errors_by_array.items()
    }

    all_errors = [
        error
        for errors in errors_by_array.values()
        for error in errors
    ]

    return {
        "clusters": len(clusters),
        "mae_by_array": mae_by_array,
        "mae_global": float(np.mean(all_errors)),
    }


def print_result(title, result, n_components=None, preserved_energy=None):
    print("=" * 80)
    print(title)
    print("=" * 80)

    if n_components is not None:
        print(f"Components: {n_components}")

    if preserved_energy is not None:
        print(f"Preserved energy: {preserved_energy:.6f}")

    print(f"Clusters: {result['clusters']}")
    print(f"MAE global: {result['mae_global']:.2f} deg")

    for array_index, mae in result["mae_by_array"].items():
        print(f"Array {array_index}: {mae:.2f} deg")

    print()


def main():
    print("=" * 80)
    print("PCA EXPERIMENT - HUMAN RANDOMWALK - WITH CRAP")
    print("=" * 80)

    start_time = time.perf_counter()

    data = DatasetCache(str(CACHE_FILE)).load()
    clutter_acquisitions = np.load(CLUTTER_FILE, allow_pickle=True)

    csi = data["csi"]
    mac = data["mac"]
    pos = data["pos"]
    timestamps = data["time"]

    print(f"Original CSI shape: {csi.shape}")
    print("Applying CRAP...")
    csi_crap = apply_crap_to_full_csi(csi, mac, clutter_acquisitions)
    print("CRAP done.")
    print()

    results = []

    baseline = evaluate_csi(
        csi=csi_crap,
        mac=mac,
        pos=pos,
        timestamps=timestamps,
    )

    print_result("BASELINE WITH CRAP", baseline)

    results.append({
        "method": "baseline_with_crap",
        "n_components": None,
        "preserved_energy": None,
        **baseline,
    })

    for variance in VARIANCES:
        csi_pca, n_components, preserved = complex_pca_reconstruct(
            csi_crap,
            variance,
        )

        result = evaluate_csi(
            csi=csi_pca,
            mac=mac,
            pos=pos,
            timestamps=timestamps,
        )

        print_result(
            f"COMPLEX PCA WITH CRAP - ENERGY {variance:.2f}",
            result,
            n_components,
            preserved,
        )

        results.append({
            "method": "complex_pca_with_crap",
            "variance_threshold": variance,
            "n_components": n_components,
            "preserved_energy": preserved,
            **result,
        })

    for variance in VARIANCES:
        csi_pca, n_components, preserved = real_imag_pca_reconstruct(
            csi_crap,
            variance,
        )

        result = evaluate_csi(
            csi=csi_pca,
            mac=mac,
            pos=pos,
            timestamps=timestamps,
        )

        print_result(
            f"REAL/IMAG PCA WITH CRAP - ENERGY {variance:.2f}",
            result,
            n_components,
            preserved,
        )

        results.append({
            "method": "real_imag_pca_with_crap",
            "variance_threshold": variance,
            "n_components": n_components,
            "preserved_energy": preserved,
            **result,
        })

    elapsed = time.perf_counter() - start_time

    output = {
        "experiment": "pca_randomwalk_human_luis_with_crap",
        "dataset": "espargos_007_human_helmet_randomwalk_1",
        "cache": str(CACHE_FILE),
        "clutter": str(CLUTTER_FILE),
        "crap": True,
        "cluster_interval_seconds": CLUSTER_INTERVAL_SECONDS,
        "variances": VARIANCES,
        "elapsed_seconds": elapsed,
        "results": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Elapsed: {elapsed:.2f} s")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()