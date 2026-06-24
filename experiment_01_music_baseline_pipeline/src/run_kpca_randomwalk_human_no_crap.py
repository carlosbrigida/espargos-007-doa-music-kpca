from pathlib import Path
import sys
import json
import time

import numpy as np
from sklearn.decomposition import PCA, KernelPCA

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.cache_dataset import DatasetCache
from src.run_uroot_music_luis_cache_with_crap import (
    cluster_cache_data,
    covariance_per_array_luis_cluster,
    get_unitary_rootmusic_estimator,
)
from src.espargos_geometry import RECEIVER_ARRAYS, true_azimuth_deg
from src.metrics import angular_error_deg


CACHE_FILE = (
    PROJECT_DIR
    / "data"
    / "cache_luis"
    / "espargos_007_human_helmet_randomwalk_1.npz"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "metrics"
    / "kpca_randomwalk_human_no_crap.json"
)

CLUSTER_INTERVAL_SECONDS = 1.0

CONFIGS = [
    ("pca", 64),
    ("pca", 128),
    ("kpca_poly", 64),
    ("kpca_poly", 128),
    ("kpca_rbf", 64),
    ("kpca_rbf", 128),
    ("kpca_cosine", 64),
    ("kpca_cosine", 128),
]


def flatten_csi(csi):
    return csi.reshape(csi.shape[0], -1)


def unflatten_csi(x, shape):
    return x.reshape(shape)


def real_imag_matrix(csi):
    x = flatten_csi(csi)
    return np.concatenate([x.real, x.imag], axis=1)


def reconstruct_pca(csi, n_components):
    shape = csi.shape

    x = real_imag_matrix(csi)

    pca = PCA(n_components=n_components)

    z = pca.fit_transform(x)
    x_hat = pca.inverse_transform(z)

    n_features = x_hat.shape[1] // 2

    real = x_hat[:, :n_features]
    imag = x_hat[:, n_features:]

    csi_hat = real + 1j * imag

    return (
        unflatten_csi(csi_hat.astype(np.complex64), shape),
        float(np.sum(pca.explained_variance_ratio_)),
    )


def reconstruct_kpca_poly(csi, n_components):
    shape = csi.shape

    x = real_imag_matrix(csi)

    kpca = KernelPCA(
        n_components=n_components,
        kernel="poly",
        degree=2,
        fit_inverse_transform=True,
        eigen_solver="arpack",
    )

    z = kpca.fit_transform(x)
    x_hat = kpca.inverse_transform(z)

    n_features = x_hat.shape[1] // 2

    real = x_hat[:, :n_features]
    imag = x_hat[:, n_features:]

    csi_hat = real + 1j * imag

    return unflatten_csi(csi_hat.astype(np.complex64), shape)


def reconstruct_kpca_rbf(csi, n_components):
    shape = csi.shape

    x = real_imag_matrix(csi)

    kpca = KernelPCA(
        n_components=n_components,
        kernel="rbf",
        gamma=1e-3,
        fit_inverse_transform=True,
        eigen_solver="arpack",
    )

    z = kpca.fit_transform(x)
    x_hat = kpca.inverse_transform(z)

    n_features = x_hat.shape[1] // 2

    real = x_hat[:, :n_features]
    imag = x_hat[:, n_features:]

    csi_hat = real + 1j * imag

    return unflatten_csi(csi_hat.astype(np.complex64), shape)


def reconstruct_kpca_cosine(csi, n_components):
    shape = csi.shape

    x = real_imag_matrix(csi)

    kpca = KernelPCA(
        n_components=n_components,
        kernel="cosine",
        fit_inverse_transform=True,
        eigen_solver="arpack",
    )

    z = kpca.fit_transform(x)
    x_hat = kpca.inverse_transform(z)

    n_features = x_hat.shape[1] // 2

    real = x_hat[:, :n_features]
    imag = x_hat[:, n_features:]

    csi_hat = real + 1j * imag

    return unflatten_csi(csi_hat.astype(np.complex64), shape)


def evaluate_clusters(clusters):
    estimator = get_unitary_rootmusic_estimator(4)

    errors_by_array = {i: [] for i in range(4)}

    for cluster in clusters:
        covariance_all = covariance_per_array_luis_cluster(
            cluster["csi_by_transmitter"]
        )

        for array_index in range(4):
            array = RECEIVER_ARRAYS[array_index]

            true_angle = true_azimuth_deg(
                array,
                cluster["mean_position"],
            )

            electrical_angle, _ = estimator(
                covariance_all[array_index]
            )

            if np.isnan(electrical_angle):
                continue

            estimated_angle = np.rad2deg(
                np.arcsin(
                    np.clip(
                        electrical_angle / np.pi,
                        -1.0,
                        1.0,
                    )
                )
            )

            error = angular_error_deg(
                estimated_angle,
                true_angle,
            )

            errors_by_array[array_index].append(error)

    mae_by_array = {
        idx: float(np.mean(errors))
        for idx, errors in errors_by_array.items()
    }

    all_errors = []

    for errors in errors_by_array.values():
        all_errors.extend(errors)

    return {
        "mae_global": float(np.mean(all_errors)),
        "mae_by_array": mae_by_array,
    }


def print_result(title, result):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(f"MAE global: {result['mae_global']:.2f} deg")

    for idx, value in result["mae_by_array"].items():
        print(f"Array {idx}: {value:.2f} deg")


def main():
    print("=" * 80)
    print("KPCA EXPERIMENT - HUMAN RANDOMWALK - NO CRAP")
    print("=" * 80)

    start = time.perf_counter()

    data = DatasetCache(str(CACHE_FILE)).load()

    csi = data["csi"]
    mac = data["mac"]
    pos = data["pos"]
    timestamps = data["time"]

    print(f"CSI shape: {csi.shape}")

    clusters = cluster_cache_data(
        csi=csi,
        mac=mac,
        pos=pos,
        timestamps=timestamps,
        cluster_interval=CLUSTER_INTERVAL_SECONDS,
    )

    baseline = evaluate_clusters(clusters)

    print_result(
        "BASELINE NO CRAP",
        baseline,
    )

    results = [
        {
            "method": "baseline",
            **baseline,
        }
    ]

    #
    # Para não explodir memória no KPCA
    #
    rng = np.random.default_rng(42)

    sample_size = min(5000, csi.shape[0])

    sample_idx = rng.choice(
        csi.shape[0],
        size=sample_size,
        replace=False,
    )

    csi_sample = csi[sample_idx]

    for method, n_components in CONFIGS:

        print()
        print(f"Running {method} ({n_components} comps)")

        t0 = time.perf_counter()

        try:

            if method == "pca":
                csi_reconstructed, energy = reconstruct_pca(
                    csi,
                    n_components,
                )

            elif method == "kpca_poly":

                kpca_csi = reconstruct_kpca_poly(
                    csi_sample,
                    n_components,
                )

                csi_reconstructed = csi.copy()

                csi_reconstructed[sample_idx] = kpca_csi

                energy = None

            elif method == "kpca_rbf":

                kpca_csi = reconstruct_kpca_rbf(
                    csi_sample,
                    n_components,
                )

                csi_reconstructed = csi.copy()

                csi_reconstructed[sample_idx] = kpca_csi

                energy = None

            elif method == "kpca_cosine":

                kpca_csi = reconstruct_kpca_cosine(
                    csi_sample,
                    n_components,
                )

                csi_reconstructed = csi.copy()

                csi_reconstructed[sample_idx] = kpca_csi

                energy = None

            else:
                continue

            pca_clusters = cluster_cache_data(
                csi=csi_reconstructed,
                mac=mac,
                pos=pos,
                timestamps=timestamps,
                cluster_interval=CLUSTER_INTERVAL_SECONDS,
            )

            result = evaluate_clusters(
                pca_clusters,
            )

            elapsed = time.perf_counter() - t0

            print_result(
                f"{method.upper()} ({n_components})",
                result,
            )

            results.append(
                {
                    "method": method,
                    "n_components": n_components,
                    "energy": energy,
                    "elapsed_seconds": elapsed,
                    **result,
                }
            )

        except Exception as exc:
            print(f"FAILED: {method} {n_components}")
            print(exc)

    total_elapsed = time.perf_counter() - start

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "experiment": "kpca_randomwalk_human_no_crap",
                "elapsed_seconds": total_elapsed,
                "results": results,
            },
            file,
            indent=4,
        )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Elapsed: {total_elapsed:.2f} s")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()