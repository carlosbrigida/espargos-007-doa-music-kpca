from pathlib import Path
import json
import sys
import time

import numpy as np
from sklearn.decomposition import PCA, KernelPCA

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.cache_dataset import DatasetCache
from src.run_uroot_music_luis_cache_with_crap import (
    remove_clutter,
    cluster_cache_data,
    covariance_per_array_luis_cluster,
    get_unitary_rootmusic_estimator,
)
from src.espargos_geometry import (
    RECEIVER_ARRAYS,
    true_azimuth_deg,
)
from src.metrics import angular_error_deg


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

SPLIT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "splits"
    / "human_randomwalk_cluster_split.json"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "metrics"
    / "kpca_r_gridsearch_train.json"
)


def build_clusters_with_crap():
    data = DatasetCache(str(CACHE_FILE)).load()

    clutter_acquisitions = np.load(
        CLUTTER_FILE,
        allow_pickle=True,
    )

    clusters = cluster_cache_data(
        csi=data["csi"],
        mac=data["mac"],
        pos=data["pos"],
        timestamps=data["time"],
        cluster_interval=1.0,
    )

    processed = []

    for cluster in clusters:

        csi_noclutter = []

        for tx_index, tx_csi in enumerate(
            cluster["csi_by_transmitter"]
        ):
            tx_csi_noclutter = remove_clutter(
                tx_csi,
                clutter_acquisitions[tx_index],
            )

            csi_noclutter.append(
                tx_csi_noclutter
            )

        processed.append(
            {
                "mean_position": cluster["mean_position"],
                "covariance": covariance_per_array_luis_cluster(
                    csi_noclutter
                ),
            }
        )

    return processed


def vectorize_covariance(R):
    return np.concatenate(
        [
            R.real.flatten(),
            R.imag.flatten(),
        ]
    )


def devectorize_covariance(v):
    real = v[:16].reshape(4, 4)
    imag = v[16:].reshape(4, 4)

    R = real + 1j * imag

    R = 0.5 * (R + R.conj().T)

    return R.astype(np.complex64)


def evaluate_covariances(
    clusters,
    reconstructed_covariances=None,
):
    estimator = get_unitary_rootmusic_estimator(4)

    errors_by_array = {
        i: []
        for i in range(4)
    }

    for cluster_index, cluster in enumerate(clusters):

        if reconstructed_covariances is None:
            covariance_all = cluster["covariance"]
        else:
            covariance_all = reconstructed_covariances[
                cluster_index
            ]

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

            errors_by_array[array_index].append(
                error
            )

    mae_by_array = {
        idx: float(np.mean(values))
        for idx, values in errors_by_array.items()
    }

    all_errors = []

    for values in errors_by_array.values():
        all_errors.extend(values)

    return {
        "mae_global": float(np.mean(all_errors)),
        "mae_by_array": mae_by_array,
    }


def reconstruct_cluster_covariances(
    reconstructed_vectors,
    n_clusters,
):
    output = []

    idx = 0

    for _ in range(n_clusters):

        arrays = []

        for _array in range(4):

            arrays.append(
                devectorize_covariance(
                    reconstructed_vectors[idx]
                )
            )

            idx += 1

        output.append(
            np.asarray(arrays)
        )

    return output


def run_pca(train_vectors, k):

    model = PCA(
        n_components=k
    )

    z = model.fit_transform(
        train_vectors
    )

    reconstructed = model.inverse_transform(z)

    return reconstructed


def run_kpca_poly(train_vectors, k, degree):

    model = KernelPCA(
        n_components=k,
        kernel="poly",
        degree=degree,
        fit_inverse_transform=True,
    )

    z = model.fit_transform(
        train_vectors
    )

    reconstructed = model.inverse_transform(z)

    return reconstructed


def run_kpca_rbf(train_vectors, k, gamma):

    model = KernelPCA(
        n_components=k,
        kernel="rbf",
        gamma=gamma,
        fit_inverse_transform=True,
    )

    z = model.fit_transform(
        train_vectors
    )

    reconstructed = model.inverse_transform(z)

    return reconstructed


def run_kpca_cosine(train_vectors, k):

    model = KernelPCA(
        n_components=k,
        kernel="cosine",
        fit_inverse_transform=True,
    )

    z = model.fit_transform(
        train_vectors
    )

    reconstructed = model.inverse_transform(z)

    return reconstructed


def main():

    print("=" * 80)
    print("KPCA(R) GRID SEARCH - TRAIN")
    print("=" * 80)

    start_time = time.perf_counter()

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        split = json.load(file)

    train_indices = split["indices"]["train"]

    clusters = build_clusters_with_crap()

    train_clusters = [
        clusters[idx]
        for idx in train_indices
    ]

    print(
        f"Train clusters: "
        f"{len(train_clusters)}"
    )

    baseline = evaluate_covariances(
        train_clusters
    )

    print(
        f"Baseline MAE: "
        f"{baseline['mae_global']:.4f} deg"
    )

    train_vectors = []

    for cluster in train_clusters:

        for array_index in range(4):

            train_vectors.append(
                vectorize_covariance(
                    cluster["covariance"][
                        array_index
                    ]
                )
            )

    train_vectors = np.asarray(
        train_vectors,
        dtype=np.float64,
    )

    results = []

    k_values = [2, 4, 6, 8, 12, 16]

    #
    # PCA
    #

    for k in k_values:

        reconstructed = run_pca(
            train_vectors,
            k,
        )

        reconstructed_cov = (
            reconstruct_cluster_covariances(
                reconstructed,
                len(train_clusters),
            )
        )

        metrics = evaluate_covariances(
            train_clusters,
            reconstructed_cov,
        )

        results.append(
            {
                "method": "pca",
                "n_components": k,
                **metrics,
            }
        )

        print(
            f"PCA k={k:<2d} "
            f"MAE={metrics['mae_global']:.4f}"
        )

    #
    # POLY
    #

    for degree in [2, 3, 4]:

        for k in k_values:

            reconstructed = run_kpca_poly(
                train_vectors,
                k,
                degree,
            )

            reconstructed_cov = (
                reconstruct_cluster_covariances(
                    reconstructed,
                    len(train_clusters),
                )
            )

            metrics = evaluate_covariances(
                train_clusters,
                reconstructed_cov,
            )

            results.append(
                {
                    "method": "kpca_poly",
                    "degree": degree,
                    "n_components": k,
                    **metrics,
                }
            )

            print(
                f"POLY d={degree} "
                f"k={k:<2d} "
                f"MAE={metrics['mae_global']:.4f}"
            )

    #
    # RBF
    #

    gammas = [
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
        1e-1,
    ]

    for gamma in gammas:

        for k in k_values:

            reconstructed = run_kpca_rbf(
                train_vectors,
                k,
                gamma,
            )

            reconstructed_cov = (
                reconstruct_cluster_covariances(
                    reconstructed,
                    len(train_clusters),
                )
            )

            metrics = evaluate_covariances(
                train_clusters,
                reconstructed_cov,
            )

            results.append(
                {
                    "method": "kpca_rbf",
                    "gamma": gamma,
                    "n_components": k,
                    **metrics,
                }
            )

            print(
                f"RBF g={gamma:.1e} "
                f"k={k:<2d} "
                f"MAE={metrics['mae_global']:.4f}"
            )

    #
    # COSINE
    #

    for k in k_values:

        reconstructed = run_kpca_cosine(
            train_vectors,
            k,
        )

        reconstructed_cov = (
            reconstruct_cluster_covariances(
                reconstructed,
                len(train_clusters),
            )
        )

        metrics = evaluate_covariances(
            train_clusters,
            reconstructed_cov,
        )

        results.append(
            {
                "method": "kpca_cosine",
                "n_components": k,
                **metrics,
            }
        )

        print(
            f"COSINE k={k:<2d} "
            f"MAE={metrics['mae_global']:.4f}"
        )

    ranking = sorted(
        results,
        key=lambda x: x["mae_global"]
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    output = {
        "baseline": baseline,
        "best_model": ranking[0],
        "top_20": ranking[:20],
        "all_results": ranking,
        "elapsed_seconds": elapsed,
    }

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
            output,
            file,
            indent=4,
        )

    print()
    print("=" * 80)
    print("TOP 10")
    print("=" * 80)

    for idx, result in enumerate(
        ranking[:10],
        start=1,
    ):
        print(
            f"{idx:2d} | "
            f"{result['method']} | "
            f"MAE={result['mae_global']:.4f}"
        )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(
        f"Elapsed: "
        f"{elapsed:.2f} s"
    )
    print(
        f"Saved: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()