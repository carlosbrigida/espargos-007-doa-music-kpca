from pathlib import Path
import json
import sys

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

    processed_clusters = []

    for cluster in clusters:

        csi_by_transmitter_noclutter = []

        for tx_index, tx_csi in enumerate(
            cluster["csi_by_transmitter"]
        ):
            tx_csi_noclutter = remove_clutter(
                tx_csi,
                clutter_acquisitions[tx_index],
            )

            csi_by_transmitter_noclutter.append(
                tx_csi_noclutter
            )

        processed_clusters.append(
            {
                "mean_position": cluster["mean_position"],
                "covariance": covariance_per_array_luis_cluster(
                    csi_by_transmitter_noclutter
                ),
            }
        )

    return processed_clusters


def vectorize_covariance(R):
    return np.concatenate(
        [
            R.real.flatten(),
            R.imag.flatten(),
        ]
    )


def devectorize_covariance(v):
    n = 16

    real = v[:n].reshape(4, 4)
    imag = v[n:].reshape(4, 4)

    R = real + 1j * imag

    R = 0.5 * (R + R.conj().T)

    return R.astype(np.complex64)


def evaluate_covariances(
    cluster_data,
    reconstructed_covariances=None,
):
    estimator = get_unitary_rootmusic_estimator(4)

    errors_by_array = {
        i: []
        for i in range(4)
    }

    for cluster_index, cluster in enumerate(cluster_data):

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


def train_pca_r(
    train_vectors,
    n_components=8,
):
    model = PCA(
        n_components=n_components
    )

    z = model.fit_transform(
        train_vectors
    )

    x_hat = model.inverse_transform(z)

    return (
        model,
        x_hat,
    )


def train_kpca_poly(
    train_vectors,
    n_components=8,
):
    model = KernelPCA(
        n_components=n_components,
        kernel="poly",
        degree=2,
        fit_inverse_transform=True,
    )

    z = model.fit_transform(
        train_vectors
    )

    x_hat = model.inverse_transform(z)

    return (
        model,
        x_hat,
    )


def train_kpca_rbf(
    train_vectors,
    n_components=8,
):
    model = KernelPCA(
        n_components=n_components,
        kernel="rbf",
        gamma=0.01,
        fit_inverse_transform=True,
    )

    z = model.fit_transform(
        train_vectors
    )

    x_hat = model.inverse_transform(z)

    return (
        model,
        x_hat,
    )


def train_kpca_cosine(
    train_vectors,
    n_components=8,
):
    model = KernelPCA(
        n_components=n_components,
        kernel="cosine",
        fit_inverse_transform=True,
    )

    z = model.fit_transform(
        train_vectors
    )

    x_hat = model.inverse_transform(z)

    return (
        model,
        x_hat,
    )


def print_result(
    title,
    result,
):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(
        f"MAE global: "
        f"{result['mae_global']:.2f} deg"
    )

    for idx, mae in result[
        "mae_by_array"
    ].items():
        print(
            f"Array {idx}: "
            f"{mae:.2f} deg"
        )


def main():

    print("=" * 80)
    print("KPCA(R) SMOKE TEST")
    print("=" * 80)

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        split = json.load(file)

    train_indices = split[
        "indices"
    ]["train"]

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

    print_result(
        "BASELINE",
        baseline,
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

    methods = [
        (
            "PCA(R)",
            train_pca_r,
        ),
        (
            "KPCA POLY",
            train_kpca_poly,
        ),
        (
            "KPCA RBF",
            train_kpca_rbf,
        ),
        (
            "KPCA COSINE",
            train_kpca_cosine,
        ),
    ]

    results = []

    for method_name, trainer in methods:

        print()
        print(
            f"Running "
            f"{method_name}"
        )

        model, reconstructed = trainer(
            train_vectors,
            n_components=8,
        )

        reconstructed_covariances = []

        idx = 0

        for _ in train_clusters:

            arrays = []

            for _array in range(4):

                arrays.append(
                    devectorize_covariance(
                        reconstructed[idx]
                    )
                )

                idx += 1

            reconstructed_covariances.append(
                np.asarray(arrays)
            )

        result = evaluate_covariances(
            train_clusters,
            reconstructed_covariances,
        )

        print_result(
            method_name,
            result,
        )

        results.append(
            {
                "method": method_name,
                **result,
            }
        )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()