from pathlib import Path
import json
import time

import numpy as np
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA

from src.run_kpca_r_smoke_test import (
    PROJECT_DIR,
    SPLIT_FILE,
    build_clusters_with_crap,
    vectorize_covariance,
    devectorize_covariance,
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "metrics"
    / "pca_r_noise_subspace_analysis.json"
)

PCA_COMPONENTS = 12
SOURCE_COUNT = 1


def build_vectors(clusters):
    vectors = []
    metadata = []

    for cluster_index, cluster in enumerate(clusters):
        for array_index in range(4):
            vectors.append(
                vectorize_covariance(
                    cluster["covariance"][array_index]
                )
            )

            metadata.append(
                (
                    cluster_index,
                    array_index,
                )
            )

    return (
        np.asarray(vectors, dtype=np.float64),
        metadata,
    )


def noise_subspace(covariance):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]

    eigenvectors = eigenvectors[:, order]

    return eigenvectors[:, SOURCE_COUNT:]


def principal_angle_deg(
    original_noise,
    reconstructed_noise,
):
    angles = subspace_angles(
        original_noise,
        reconstructed_noise,
    )

    return np.rad2deg(angles)


def main():

    print("=" * 80)
    print("PCA(R) NOISE SUBSPACE ANALYSIS")
    print("=" * 80)

    start = time.perf_counter()

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        split = json.load(file)

    train_indices = split["indices"]["train"]
    test_indices = split["indices"]["test"]

    clusters = build_clusters_with_crap()

    train_clusters = [
        clusters[idx]
        for idx in train_indices
    ]

    test_clusters = [
        clusters[idx]
        for idx in test_indices
    ]

    train_vectors, _ = build_vectors(
        train_clusters
    )

    test_vectors, metadata = build_vectors(
        test_clusters
    )

    model = PCA(
        n_components=PCA_COMPONENTS
    )

    model.fit(train_vectors)

    reconstructed_vectors = (
        model.inverse_transform(
            model.transform(test_vectors)
        )
    )

    max_angles = []
    mean_angles = []

    for row_index, vector in enumerate(
        reconstructed_vectors
    ):

        cluster_index, array_index = (
            metadata[row_index]
        )

        r_original = (
            test_clusters[cluster_index]
            ["covariance"][array_index]
        )

        r_reconstructed = (
            devectorize_covariance(vector)
        )

        noise_original = noise_subspace(
            r_original
        )

        noise_reconstructed = noise_subspace(
            r_reconstructed
        )

        angles = principal_angle_deg(
            noise_original,
            noise_reconstructed,
        )

        max_angles.append(
            float(np.max(angles))
        )

        mean_angles.append(
            float(np.mean(angles))
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    result = {
        "experiment":
            "pca_r_noise_subspace_analysis",
        "split":
            "test",
        "pca_components":
            PCA_COMPONENTS,
        "source_count":
            SOURCE_COUNT,
        "max_principal_angle_deg": {
            "mean":
                float(np.mean(max_angles)),
            "median":
                float(np.median(max_angles)),
            "max":
                float(np.max(max_angles)),
        },
        "mean_principal_angle_deg": {
            "mean":
                float(np.mean(mean_angles)),
            "median":
                float(np.median(mean_angles)),
            "max":
                float(np.max(mean_angles)),
        },
        "elapsed_seconds":
            elapsed,
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
            result,
            file,
            indent=4,
        )

    print()
    print("Mean principal angle:")
    print(
        f"{result['mean_principal_angle_deg']['mean']:.4f} deg"
    )

    print()

    print("Median principal angle:")
    print(
        f"{result['mean_principal_angle_deg']['median']:.4f} deg"
    )

    print()

    print("Worst-case principal angle:")
    print(
        f"{result['max_principal_angle_deg']['max']:.4f} deg"
    )

    print()

    print(f"Saved: {OUTPUT_FILE}")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()