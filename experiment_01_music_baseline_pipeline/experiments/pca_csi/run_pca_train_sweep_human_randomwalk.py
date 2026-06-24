from pathlib import Path
import sys
import json
import time

import numpy as np
from sklearn.decomposition import PCA

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.cache_dataset import DatasetCache
from src.run_uroot_music_luis_cache_with_crap import (
    remove_clutter,
    cluster_cache_data,
    covariance_per_array_luis_cluster,
    get_unitary_rootmusic_estimator,
)
from src.espargos_geometry import RECEIVER_ARRAYS, true_azimuth_deg
from src.metrics import angular_error_deg


CACHE_FILE = PROJECT_DIR / "data/cache_luis/espargos_007_human_helmet_randomwalk_1.npz"
CLUTTER_FILE = PROJECT_DIR / "clutter_channel_estimates_luis/espargos_007_human_helmet_randomwalk_1.npz.npy"
SPLIT_FILE = PROJECT_DIR / "outputs/splits/human_randomwalk_cluster_split.json"

OUTPUT_FILE = PROJECT_DIR / "outputs/metrics/pca_train_sweep_human_randomwalk.json"

CLUSTER_INTERVAL_SECONDS = 1.0
VARIANCES = [0.80, 0.85, 0.90, 0.95, 0.97, 0.99]


def apply_crap(csi, mac, clutter_acquisitions):
    csi_out = np.empty_like(csi)
    unique_macs = np.unique(mac)

    for tx_index, current_mac in enumerate(unique_macs):
        mask = mac == current_mac
        csi_out[mask] = remove_clutter(
            csi[mask],
            clutter_acquisitions[tx_index],
        )

    return csi_out


def flatten_csi(csi):
    return csi.reshape(csi.shape[0], -1)


def unflatten_csi(x, shape):
    return x.reshape(shape)


def complex_pca_reconstruct(csi, variance):
    shape = csi.shape
    x = flatten_csi(csi)

    mean = np.mean(x, axis=0, keepdims=True)
    x_centered = x - mean

    u, s, vh = np.linalg.svd(x_centered, full_matrices=False)

    energy = s ** 2
    cumulative = np.cumsum(energy) / np.sum(energy)

    n_components = int(np.searchsorted(cumulative, variance) + 1)

    x_hat = (
        u[:, :n_components]
        @ np.diag(s[:n_components])
        @ vh[:n_components, :]
    ) + mean

    return (
        unflatten_csi(x_hat.astype(np.complex64), shape),
        n_components,
        float(cumulative[n_components - 1]),
    )


def real_imag_pca_reconstruct(csi, variance):
    shape = csi.shape
    x = flatten_csi(csi)

    x_realimag = np.concatenate([x.real, x.imag], axis=1)

    pca = PCA(n_components=variance, svd_solver="full")
    z = pca.fit_transform(x_realimag)
    x_hat_realimag = pca.inverse_transform(z)

    n_features = x.shape[1]

    real = x_hat_realimag[:, :n_features]
    imag = x_hat_realimag[:, n_features:]

    x_hat = real + 1j * imag

    return (
        unflatten_csi(x_hat.astype(np.complex64), shape),
        int(pca.n_components_),
        float(np.sum(pca.explained_variance_ratio_)),
    )


def evaluate_clusters(clusters):
    estimator = get_unitary_rootmusic_estimator(4)
    errors_by_array = {i: [] for i in range(4)}

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

            estimated_angle_deg = np.rad2deg(
                np.arcsin(np.clip(electrical_angle / np.pi, -1.0, 1.0))
            )

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
        "mae_by_array": mae_by_array,
        "mae_global": float(np.mean(all_errors)),
    }


def select_clusters(clusters, indices):
    return [clusters[index] for index in indices]


def compression_ratio(method, n_components, original_complex_features):
    if method == "complex":
        return original_complex_features / n_components

    if method == "real_imag":
        return (2 * original_complex_features) / n_components

    return 1.0


def main():
    print("=" * 80)
    print("PCA TRAIN SWEEP - HUMAN RANDOMWALK")
    print("=" * 80)

    start = time.perf_counter()

    data = DatasetCache(str(CACHE_FILE)).load()
    clutter = np.load(CLUTTER_FILE, allow_pickle=True)

    with open(SPLIT_FILE, "r", encoding="utf-8") as file:
        split = json.load(file)

    csi = data["csi"]
    mac = data["mac"]
    pos = data["pos"]
    timestamps = data["time"]

    print(f"CSI shape: {csi.shape}")
    print("Applying CRAP...")
    csi_crap = apply_crap(csi, mac, clutter)
    print("CRAP done.")

    clusters = cluster_cache_data(
        csi=csi_crap,
        mac=mac,
        pos=pos,
        timestamps=timestamps,
        cluster_interval=CLUSTER_INTERVAL_SECONDS,
    )

    train_indices = split["indices"]["train"]
    train_clusters = select_clusters(clusters, train_indices)

    print(f"Total clusters: {len(clusters)}")
    print(f"Train clusters: {len(train_clusters)}")
    print()

    original_complex_features = int(np.prod(csi.shape[1:]))

    results = []

    baseline = evaluate_clusters(train_clusters)

    print("=" * 80)
    print("TRAIN BASELINE WITH CRAP")
    print("=" * 80)
    print(f"MAE global: {baseline['mae_global']:.2f} deg")
    print()

    results.append({
        "split": "train",
        "method": "baseline_with_crap",
        "variance_threshold": None,
        "n_components": None,
        "preserved_energy": None,
        "compression_ratio": 1.0,
        **baseline,
    })

    for variance in VARIANCES:
        print("=" * 80)
        print(f"COMPLEX PCA | TRAIN | ENERGY {variance:.2f}")
        print("=" * 80)

        csi_pca, n_components, preserved = complex_pca_reconstruct(
            csi_crap,
            variance,
        )

        pca_clusters = cluster_cache_data(
            csi=csi_pca,
            mac=mac,
            pos=pos,
            timestamps=timestamps,
            cluster_interval=CLUSTER_INTERVAL_SECONDS,
        )

        train_pca_clusters = select_clusters(pca_clusters, train_indices)
        result = evaluate_clusters(train_pca_clusters)

        ratio = compression_ratio(
            "complex",
            n_components,
            original_complex_features,
        )

        print(f"Components: {n_components}")
        print(f"Preserved energy: {preserved:.6f}")
        print(f"Compression ratio: {ratio:.2f}x")
        print(f"MAE global: {result['mae_global']:.2f} deg")
        print()

        results.append({
            "split": "train",
            "method": "complex_pca_with_crap",
            "variance_threshold": variance,
            "n_components": n_components,
            "preserved_energy": preserved,
            "compression_ratio": ratio,
            **result,
        })

    for variance in VARIANCES:
        print("=" * 80)
        print(f"REAL/IMAG PCA | TRAIN | ENERGY {variance:.2f}")
        print("=" * 80)

        csi_pca, n_components, preserved = real_imag_pca_reconstruct(
            csi_crap,
            variance,
        )

        pca_clusters = cluster_cache_data(
            csi=csi_pca,
            mac=mac,
            pos=pos,
            timestamps=timestamps,
            cluster_interval=CLUSTER_INTERVAL_SECONDS,
        )

        train_pca_clusters = select_clusters(pca_clusters, train_indices)
        result = evaluate_clusters(train_pca_clusters)

        ratio = compression_ratio(
            "real_imag",
            n_components,
            original_complex_features,
        )

        print(f"Components: {n_components}")
        print(f"Preserved energy: {preserved:.6f}")
        print(f"Compression ratio: {ratio:.2f}x")
        print(f"MAE global: {result['mae_global']:.2f} deg")
        print()

        results.append({
            "split": "train",
            "method": "real_imag_pca_with_crap",
            "variance_threshold": variance,
            "n_components": n_components,
            "preserved_energy": preserved,
            "compression_ratio": ratio,
            **result,
        })

    elapsed = time.perf_counter() - start

    output = {
        "experiment": "pca_train_sweep_human_randomwalk",
        "dataset": "espargos_007_human_helmet_randomwalk_1",
        "split_file": str(SPLIT_FILE),
        "split": "train",
        "crap": True,
        "cluster_interval_seconds": CLUSTER_INTERVAL_SECONDS,
        "variances": VARIANCES,
        "original_complex_features": original_complex_features,
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