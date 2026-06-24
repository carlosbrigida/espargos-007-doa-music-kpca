from pathlib import Path
import json
import time
import warnings

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
    / "kpca_r_gridsearch_train_robust.json"
)


K_VALUES = [2, 4, 6, 8, 12, 16]

RBF_GAMMAS = [
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
]

POLY_DEGREES = [2, 3, 4]


def reconstruct_cluster_covariances(reconstructed_vectors, n_clusters):
    output = []
    idx = 0

    for _ in range(n_clusters):
        arrays = []

        for _array in range(4):
            arrays.append(devectorize_covariance(reconstructed_vectors[idx]))
            idx += 1

        output.append(np.asarray(arrays))

    return output


def build_train_vectors(train_clusters):
    vectors = []

    for cluster in train_clusters:
        for array_index in range(4):
            vectors.append(
                vectorize_covariance(cluster["covariance"][array_index])
            )

    return np.asarray(vectors, dtype=np.float64)


def run_model(train_vectors, config):
    method = config["method"]
    k = config["n_components"]

    if method == "pca":
        model = PCA(n_components=k)

    elif method == "kpca_poly":
        model = KernelPCA(
            n_components=k,
            kernel="poly",
            degree=config["degree"],
            fit_inverse_transform=True,
            alpha=1e-3,
        )

    elif method == "kpca_rbf":
        model = KernelPCA(
            n_components=k,
            kernel="rbf",
            gamma=config["gamma"],
            fit_inverse_transform=True,
            alpha=1e-3,
        )

    elif method == "kpca_cosine":
        model = KernelPCA(
            n_components=k,
            kernel="cosine",
            fit_inverse_transform=True,
            alpha=1e-3,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    transformed = model.fit_transform(train_vectors)
    reconstructed = model.inverse_transform(transformed)

    return reconstructed


def make_configs():
    configs = []

    for k in K_VALUES:
        configs.append(
            {
                "method": "pca",
                "n_components": k,
            }
        )

    for degree in POLY_DEGREES:
        for k in K_VALUES:
            configs.append(
                {
                    "method": "kpca_poly",
                    "degree": degree,
                    "n_components": k,
                }
            )

    for gamma in RBF_GAMMAS:
        for k in K_VALUES:
            configs.append(
                {
                    "method": "kpca_rbf",
                    "gamma": gamma,
                    "n_components": k,
                }
            )

    for k in K_VALUES:
        configs.append(
            {
                "method": "kpca_cosine",
                "n_components": k,
            }
        )

    return configs


def describe_config(config):
    method = config["method"]
    k = config["n_components"]

    if method == "kpca_poly":
        return f"{method} degree={config['degree']} k={k}"

    if method == "kpca_rbf":
        return f"{method} gamma={config['gamma']:.1e} k={k}"

    return f"{method} k={k}"


def main():
    print("=" * 80)
    print("KPCA(R) GRID SEARCH TRAIN - ROBUST")
    print("=" * 80)

    start_time = time.perf_counter()

    with open(SPLIT_FILE, "r", encoding="utf-8") as file:
        split = json.load(file)

    train_indices = split["indices"]["train"]

    clusters = build_clusters_with_crap()
    train_clusters = [clusters[idx] for idx in train_indices]

    print(f"Train clusters: {len(train_clusters)}")

    baseline = evaluate_covariances(train_clusters)

    print(f"Baseline MAE: {baseline['mae_global']:.4f} deg")

    train_vectors = build_train_vectors(train_clusters)

    configs = make_configs()

    results = []
    failures = []

    for i, config in enumerate(configs, start=1):
        label = describe_config(config)

        print(f"[{i:03d}/{len(configs):03d}] Running {label}")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                reconstructed = run_model(train_vectors, config)

            reconstructed_covariances = reconstruct_cluster_covariances(
                reconstructed,
                len(train_clusters),
            )

            metrics = evaluate_covariances(
                train_clusters,
                reconstructed_covariances,
            )

            result = {
                **config,
                **metrics,
            }

            results.append(result)

            print(f"    MAE = {metrics['mae_global']:.4f} deg")

        except Exception as exc:
            failure = {
                **config,
                "error": str(exc),
            }

            failures.append(failure)

            print(f"    FAILED: {exc}")

    ranking = sorted(results, key=lambda item: item["mae_global"])

    elapsed = time.perf_counter() - start_time

    output = {
        "experiment": "kpca_r_gridsearch_train_robust",
        "baseline": baseline,
        "best_model": ranking[0] if ranking else None,
        "top_20": ranking[:20],
        "all_results": ranking,
        "failures": failures,
        "elapsed_seconds": elapsed,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print()
    print("=" * 80)
    print("TOP 10")
    print("=" * 80)

    for idx, result in enumerate(ranking[:10], start=1):
        print(
            f"{idx:02d} | "
            f"{describe_config(result)} | "
            f"MAE={result['mae_global']:.4f}"
        )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Failures: {len(failures)}")
    print(f"Elapsed: {elapsed:.2f} s")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()