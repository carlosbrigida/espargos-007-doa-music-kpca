from pathlib import Path
import json
import time

import numpy as np
from sklearn.decomposition import PCA

from src.run_kpca_r_smoke_test import (
    PROJECT_DIR,
    SPLIT_FILE,
    build_clusters_with_crap,
    vectorize_covariance,
    devectorize_covariance,
)

OUTPUT_FILE = PROJECT_DIR / "outputs" / "metrics" / "pca_r_physics_analysis.json"

PCA_COMPONENTS = 12


def build_vectors(clusters):
    vectors = []
    metadata = []

    for cluster_index, cluster in enumerate(clusters):
        for array_index in range(4):
            vectors.append(vectorize_covariance(cluster["covariance"][array_index]))
            metadata.append((cluster_index, array_index))

    return np.asarray(vectors, dtype=np.float64), metadata


def frobenius_relative_error(r_original, r_reconstructed):
    numerator = np.linalg.norm(r_original - r_reconstructed, ord="fro")
    denominator = np.linalg.norm(r_original, ord="fro")

    if denominator == 0:
        return np.nan

    return float(numerator / denominator)


def hermitian_error(r_matrix):
    return float(
        np.linalg.norm(r_matrix - r_matrix.conj().T, ord="fro")
        / max(np.linalg.norm(r_matrix, ord="fro"), 1e-12)
    )


def eigenvalue_difference(r_original, r_reconstructed):
    eig_original = np.linalg.eigvalsh(r_original)
    eig_reconstructed = np.linalg.eigvalsh(r_reconstructed)

    return float(
        np.linalg.norm(eig_original - eig_reconstructed)
        / max(np.linalg.norm(eig_original), 1e-12)
    )


def main():
    print("=" * 80)
    print("PCA(R) PHYSICS ANALYSIS")
    print("=" * 80)

    start = time.perf_counter()

    with open(SPLIT_FILE, "r", encoding="utf-8") as file:
        split = json.load(file)

    train_indices = split["indices"]["train"]
    test_indices = split["indices"]["test"]

    clusters = build_clusters_with_crap()

    train_clusters = [clusters[index] for index in train_indices]
    test_clusters = [clusters[index] for index in test_indices]

    train_vectors, _ = build_vectors(train_clusters)
    test_vectors, test_metadata = build_vectors(test_clusters)

    model = PCA(n_components=PCA_COMPONENTS)
    model.fit(train_vectors)

    reconstructed_vectors = model.inverse_transform(
        model.transform(test_vectors)
    )

    fro_errors = []
    eig_errors = []
    herm_errors = []
    min_eigenvalues = []

    for row_index, reconstructed_vector in enumerate(reconstructed_vectors):
        cluster_index, array_index = test_metadata[row_index]

        r_original = test_clusters[cluster_index]["covariance"][array_index]
        r_reconstructed = devectorize_covariance(reconstructed_vector)

        fro_errors.append(
            frobenius_relative_error(r_original, r_reconstructed)
        )

        eig_errors.append(
            eigenvalue_difference(r_original, r_reconstructed)
        )

        herm_errors.append(
            hermitian_error(r_reconstructed)
        )

        min_eigenvalues.append(
            float(np.min(np.linalg.eigvalsh(r_reconstructed)))
        )

    result = {
        "experiment": "pca_r_physics_analysis",
        "split": "test",
        "pca_components": PCA_COMPONENTS,
        "frobenius_relative_error": {
            "mean": float(np.mean(fro_errors)),
            "median": float(np.median(fro_errors)),
            "max": float(np.max(fro_errors)),
        },
        "eigenvalue_relative_error": {
            "mean": float(np.mean(eig_errors)),
            "median": float(np.median(eig_errors)),
            "max": float(np.max(eig_errors)),
        },
        "hermitian_relative_error": {
            "mean": float(np.mean(herm_errors)),
            "median": float(np.median(herm_errors)),
            "max": float(np.max(herm_errors)),
        },
        "minimum_eigenvalue_reconstructed": {
            "mean": float(np.mean(min_eigenvalues)),
            "min": float(np.min(min_eigenvalues)),
            "negative_count": int(np.sum(np.asarray(min_eigenvalues) < -1e-6)),
        },
        "elapsed_seconds": time.perf_counter() - start,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print(f"Frobenius relative error mean: {result['frobenius_relative_error']['mean']:.6f}")
    print(f"Eigenvalue relative error mean: {result['eigenvalue_relative_error']['mean']:.6f}")
    print(f"Hermitian relative error mean: {result['hermitian_relative_error']['mean']:.6e}")
    print(f"Minimum eigenvalue: {result['minimum_eigenvalue_reconstructed']['min']:.6e}")
    print(f"Negative eigenvalue count: {result['minimum_eigenvalue_reconstructed']['negative_count']}")
    print(f"Saved: {OUTPUT_FILE}")
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()