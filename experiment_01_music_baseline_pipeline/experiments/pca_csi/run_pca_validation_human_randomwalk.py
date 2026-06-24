from src.run_pca_train_sweep_human_randomwalk import *

VALIDATION_VARIANCE = 0.95
OUTPUT_FILE = PROJECT_DIR / "outputs/metrics/pca_validation_human_randomwalk.json"


def main():
    print("=" * 80)
    print("PCA VALIDATION - HUMAN RANDOMWALK")
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

    validation_indices = split["indices"]["validation"]
    validation_clusters = select_clusters(clusters, validation_indices)

    print(f"Validation clusters: {len(validation_clusters)}")
    print()

    original_complex_features = int(np.prod(csi.shape[1:]))
    results = []

    baseline = evaluate_clusters(validation_clusters)

    print("=" * 80)
    print("VALIDATION BASELINE WITH CRAP")
    print("=" * 80)
    print(f"MAE global: {baseline['mae_global']:.2f} deg")
    print()

    results.append({
        "split": "validation",
        "method": "baseline_with_crap",
        "variance_threshold": None,
        "n_components": None,
        "preserved_energy": None,
        "compression_ratio": 1.0,
        **baseline,
    })

    print("=" * 80)
    print("VALIDATION COMPLEX PCA WITH CRAP - ENERGY 0.95")
    print("=" * 80)

    csi_complex, n_complex, preserved_complex = complex_pca_reconstruct(
        csi_crap,
        VALIDATION_VARIANCE,
    )

    complex_clusters = cluster_cache_data(
        csi=csi_complex,
        mac=mac,
        pos=pos,
        timestamps=timestamps,
        cluster_interval=CLUSTER_INTERVAL_SECONDS,
    )

    complex_validation_clusters = select_clusters(
        complex_clusters,
        validation_indices,
    )

    complex_result = evaluate_clusters(complex_validation_clusters)
    complex_ratio = compression_ratio(
        "complex",
        n_complex,
        original_complex_features,
    )

    print(f"Components: {n_complex}")
    print(f"Preserved energy: {preserved_complex:.6f}")
    print(f"Compression ratio: {complex_ratio:.2f}x")
    print(f"MAE global: {complex_result['mae_global']:.2f} deg")
    print()

    results.append({
        "split": "validation",
        "method": "complex_pca_with_crap",
        "variance_threshold": VALIDATION_VARIANCE,
        "n_components": n_complex,
        "preserved_energy": preserved_complex,
        "compression_ratio": complex_ratio,
        **complex_result,
    })

    print("=" * 80)
    print("VALIDATION REAL/IMAG PCA WITH CRAP - ENERGY 0.95")
    print("=" * 80)

    csi_realimag, n_realimag, preserved_realimag = real_imag_pca_reconstruct(
        csi_crap,
        VALIDATION_VARIANCE,
    )

    realimag_clusters = cluster_cache_data(
        csi=csi_realimag,
        mac=mac,
        pos=pos,
        timestamps=timestamps,
        cluster_interval=CLUSTER_INTERVAL_SECONDS,
    )

    realimag_validation_clusters = select_clusters(
        realimag_clusters,
        validation_indices,
    )

    realimag_result = evaluate_clusters(realimag_validation_clusters)
    realimag_ratio = compression_ratio(
        "real_imag",
        n_realimag,
        original_complex_features,
    )

    print(f"Components: {n_realimag}")
    print(f"Preserved energy: {preserved_realimag:.6f}")
    print(f"Compression ratio: {realimag_ratio:.2f}x")
    print(f"MAE global: {realimag_result['mae_global']:.2f} deg")
    print()

    results.append({
        "split": "validation",
        "method": "real_imag_pca_with_crap",
        "variance_threshold": VALIDATION_VARIANCE,
        "n_components": n_realimag,
        "preserved_energy": preserved_realimag,
        "compression_ratio": realimag_ratio,
        **realimag_result,
    })

    elapsed = time.perf_counter() - start

    output = {
        "experiment": "pca_validation_human_randomwalk",
        "dataset": "espargos_007_human_helmet_randomwalk_1",
        "split": "validation",
        "split_file": str(SPLIT_FILE),
        "crap": True,
        "variance_threshold": VALIDATION_VARIANCE,
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