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
    evaluate_covariances,
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "metrics"
    / "pca_r_test.json"
)


def build_vectors(clusters):
    vectors = []

    for cluster in clusters:
        for array_index in range(4):
            vectors.append(
                vectorize_covariance(
                    cluster["covariance"][array_index]
                )
            )

    return np.asarray(vectors, dtype=np.float64)


def reconstruct_cluster_covariances(vectors, n_clusters):
    output = []
    idx = 0

    for _ in range(n_clusters):
        arrays = []

        for _array in range(4):
            arrays.append(
                devectorize_covariance(
                    vectors[idx]
                )
            )
            idx += 1

        output.append(np.asarray(arrays))

    return output


def main():

    print("=" * 80)
    print("PCA(R) TEST")
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

    print(
        f"Train clusters: {len(train_clusters)}"
    )
    print(
        f"Test clusters: {len(test_clusters)}"
    )
    print()

    baseline = evaluate_covariances(
        test_clusters
    )

    print("=" * 80)
    print("BASELINE + CRAP")
    print("=" * 80)
    print(
        f"MAE global: "
        f"{baseline['mae_global']:.4f} deg"
    )

    for array_index, mae in baseline[
        "mae_by_array"
    ].items():
        print(
            f"Array {array_index}: "
            f"{mae:.4f} deg"
        )

    print()

    train_vectors = build_vectors(
        train_clusters
    )

    test_vectors = build_vectors(
        test_clusters
    )

    model = PCA(
        n_components=12
    )

    model.fit(train_vectors)

    z_test = model.transform(
        test_vectors
    )

    reconstructed_test = (
        model.inverse_transform(
            z_test
        )
    )

    reconstructed_covariances = (
        reconstruct_cluster_covariances(
            reconstructed_test,
            len(test_clusters),
        )
    )

    pca_result = evaluate_covariances(
        test_clusters,
        reconstructed_covariances,
    )

    print("=" * 80)
    print("PCA(R) k=12")
    print("=" * 80)
    print(
        f"MAE global: "
        f"{pca_result['mae_global']:.4f} deg"
    )

    for array_index, mae in pca_result[
        "mae_by_array"
    ].items():
        print(
            f"Array {array_index}: "
            f"{mae:.4f} deg"
        )

    gain = (
        baseline["mae_global"]
        - pca_result["mae_global"]
    )

    gain_percent = (
        100.0 * gain
        / baseline["mae_global"]
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    output = {
        "experiment": "pca_r_test",
        "split": "test",
        "baseline": baseline,
        "pca_r_k12": pca_result,
        "gain_deg": gain,
        "gain_percent": gain_percent,
        "compression_ratio": 32 / 12,
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
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Gain: {gain:.4f} deg"
    )

    print(
        f"Gain (%): "
        f"{gain_percent:.2f}%"
    )

    print(
        f"Compression: "
        f"{32/12:.2f}x"
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"Elapsed: {elapsed:.2f} s"
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()