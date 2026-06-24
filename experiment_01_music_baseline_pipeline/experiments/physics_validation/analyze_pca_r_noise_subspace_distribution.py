import json
import time
import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles

from src.run_kpca_r_smoke_test import (
    PROJECT_DIR,
    SPLIT_FILE,
    build_clusters_with_crap,
    vectorize_covariance,
    devectorize_covariance,
)

OUTPUT_FILE = PROJECT_DIR / "outputs" / "metrics" / "pca_r_noise_subspace_distribution.json"

PCA_COMPONENTS = 12
SOURCE_COUNT = 1


def build_vectors(clusters):
    vectors = []
    metadata = []

    for cluster_index, cluster in enumerate(clusters):
        for array_index in range(4):
            vectors.append(vectorize_covariance(cluster["covariance"][array_index]))
            metadata.append((cluster_index, array_index))

    return np.asarray(vectors, dtype=np.float64), metadata


def noise_subspace(R):
    eigvals, eigvecs = np.linalg.eigh(R)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    return eigvecs[:, SOURCE_COUNT:]


def main():
    print("=" * 80)
    print("PCA(R) NOISE SUBSPACE DISTRIBUTION")
    print("=" * 80)

    start = time.perf_counter()

    with open(SPLIT_FILE, "r", encoding="utf-8") as file:
        split = json.load(file)

    train_indices = split["indices"]["train"]
    test_indices = split["indices"]["test"]

    clusters = build_clusters_with_crap()

    train_clusters = [clusters[i] for i in train_indices]
    test_clusters = [clusters[i] for i in test_indices]

    train_vectors, _ = build_vectors(train_clusters)
    test_vectors, metadata = build_vectors(test_clusters)

    model = PCA(n_components=PCA_COMPONENTS)
    model.fit(train_vectors)

    reconstructed_vectors = model.inverse_transform(model.transform(test_vectors))

    mean_angles = []
    max_angles = []

    for row_index, vector in enumerate(reconstructed_vectors):
        cluster_index, array_index = metadata[row_index]

        R_original = test_clusters[cluster_index]["covariance"][array_index]
        R_reconstructed = devectorize_covariance(vector)

        angles = np.rad2deg(
            subspace_angles(
                noise_subspace(R_original),
                noise_subspace(R_reconstructed),
            )
        )

        mean_angles.append(float(np.mean(angles)))
        max_angles.append(float(np.max(angles)))

    mean_angles = np.asarray(mean_angles)
    max_angles = np.asarray(max_angles)

    bins = [0, 1, 2, 3, 4, 5, 10, 20, 40, 90]
    histogram, bin_edges = np.histogram(mean_angles, bins=bins)

    result = {
        "experiment": "pca_r_noise_subspace_distribution",
        "split": "test",
        "pca_components": PCA_COMPONENTS,
        "samples": int(len(mean_angles)),
        "mean_principal_angle_deg": {
            "mean": float(np.mean(mean_angles)),
            "median": float(np.median(mean_angles)),
            "p90": float(np.percentile(mean_angles, 90)),
            "p95": float(np.percentile(mean_angles, 95)),
            "p99": float(np.percentile(mean_angles, 99)),
            "max": float(np.max(mean_angles)),
        },
        "max_principal_angle_deg": {
            "mean": float(np.mean(max_angles)),
            "median": float(np.median(max_angles)),
            "p90": float(np.percentile(max_angles, 90)),
            "p95": float(np.percentile(max_angles, 95)),
            "p99": float(np.percentile(max_angles, 99)),
            "max": float(np.max(max_angles)),
        },
        "histogram_mean_angle": {
            "bins": bins,
            "counts": histogram.tolist(),
        },
        "elapsed_seconds": time.perf_counter() - start,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print("Mean principal angle distribution:")
    print(f"Mean:   {result['mean_principal_angle_deg']['mean']:.4f} deg")
    print(f"Median: {result['mean_principal_angle_deg']['median']:.4f} deg")
    print(f"P90:    {result['mean_principal_angle_deg']['p90']:.4f} deg")
    print(f"P95:    {result['mean_principal_angle_deg']['p95']:.4f} deg")
    print(f"P99:    {result['mean_principal_angle_deg']['p99']:.4f} deg")
    print(f"Max:    {result['mean_principal_angle_deg']['max']:.4f} deg")

    print()
    print("Histogram:")
    for i, count in enumerate(histogram):
        print(f"{bins[i]}-{bins[i+1]} deg: {count}")

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()