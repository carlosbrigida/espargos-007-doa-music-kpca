from pathlib import Path
import json
import time

import numpy as np
from sklearn.decomposition import PCA, KernelPCA

from src.run_kpca_r_smoke_test import (
    PROJECT_DIR,
    SPLIT_FILE,
    build_clusters_with_crap,
    vectorize_covariance,
    devectorize_covariance,
    evaluate_covariances,
)


OUTPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "metrics"
    / "kpca_r_validation.json"
)


def build_vectors(clusters):
    vectors = []

    for cluster in clusters:
        for array_index in range(4):
            vectors.append(
                vectorize_covariance(cluster["covariance"][array_index])
            )

    return np.asarray(vectors, dtype=np.float64)


def reconstruct_cluster_covariances(vectors, n_clusters):
    output = []
    index = 0

    for _ in range(n_clusters):
        arrays = []

        for _array in range(4):
            arrays.append(devectorize_covariance(vectors[index]))
            index += 1

        output.append(np.asarray(arrays))

    return output


def run_model(name, model, train_vectors, validation_vectors, validation_clusters):
    print("=" * 80)
    print(name)
    print("=" * 80)

    z_train = model.fit_transform(train_vectors)
    _ = model.inverse_transform(z_train)

    z_validation = model.transform(validation_vectors)
    reconstructed_validation = model.inverse_transform(z_validation)

    reconstructed_covariances = reconstruct_cluster_covariances(
        reconstructed_validation,
        len(validation_clusters),
    )

    result = evaluate_covariances(
        validation_clusters,
        reconstructed_covariances,
    )

    print(f"MAE global: {result['mae_global']:.4f} deg")

    for array_index, mae in result["mae_by_array"].items():
        print(f"Array {array_index}: {mae:.4f} deg")

    print()

    return result


def main():
    print("=" * 80)
    print("KPCA(R) VALIDATION")
    print("=" * 80)

    start = time.perf_counter()

    with open(SPLIT_FILE, "r", encoding="utf-8") as file:
        split = json.load(file)

    train_indices = split["indices"]["train"]
    validation_indices = split["indices"]["validation"]

    clusters = build_clusters_with_crap()

    train_clusters = [clusters[index] for index in train_indices]
    validation_clusters = [clusters[index] for index in validation_indices]

    print(f"Train clusters: {len(train_clusters)}")
    print(f"Validation clusters: {len(validation_clusters)}")
    print()

    train_vectors = build_vectors(train_clusters)
    validation_vectors = build_vectors(validation_clusters)

    baseline = evaluate_covariances(validation_clusters)

    print("=" * 80)
    print("BASELINE + CRAP")
    print("=" * 80)
    print(f"MAE global: {baseline['mae_global']:.4f} deg")

    for array_index, mae in baseline["mae_by_array"].items():
        print(f"Array {array_index}: {mae:.4f} deg")

    print()

    pca_model = PCA(n_components=12)

    pca_result = run_model(
        name="PCA(R) k=12",
        model=pca_model,
        train_vectors=train_vectors,
        validation_vectors=validation_vectors,
        validation_clusters=validation_clusters,
    )

    kpca_model = KernelPCA(
        n_components=8,
        kernel="poly",
        degree=3,
        fit_inverse_transform=True,
        alpha=1e-3,
    )

    kpca_result = run_model(
        name="KPCA(R) Poly degree=3 k=8",
        model=kpca_model,
        train_vectors=train_vectors,
        validation_vectors=validation_vectors,
        validation_clusters=validation_clusters,
    )

    elapsed = time.perf_counter() - start

    output = {
        "experiment": "kpca_r_validation",
        "split": "validation",
        "train_clusters": len(train_clusters),
        "validation_clusters": len(validation_clusters),
        "baseline": baseline,
        "pca_r_k12": pca_result,
        "kpca_r_poly_degree3_k8": kpca_result,
        "elapsed_seconds": elapsed,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Elapsed: {elapsed:.2f} s")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()