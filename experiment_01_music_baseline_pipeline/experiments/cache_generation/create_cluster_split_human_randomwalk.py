from pathlib import Path
import sys
import json
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.cache_dataset import DatasetCache
from src.run_uroot_music_luis_cache_with_crap import cluster_cache_data


CACHE_FILE = (
    PROJECT_DIR
    / "data"
    / "cache_luis"
    / "espargos_007_human_helmet_randomwalk_1.npz"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "splits"
    / "human_randomwalk_cluster_split.json"
)

CLUSTER_INTERVAL_SECONDS = 1.0

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20

RANDOM_SEED = 42


def main():
    data = DatasetCache(str(CACHE_FILE)).load()

    clusters = cluster_cache_data(
        csi=data["csi"],
        mac=data["mac"],
        pos=data["pos"],
        timestamps=data["time"],
        cluster_interval=CLUSTER_INTERVAL_SECONDS,
    )

    n_clusters = len(clusters)

    indices = np.arange(n_clusters)

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(indices)

    train_end = int(TRAIN_RATIO * n_clusters)
    validation_end = int((TRAIN_RATIO + VALIDATION_RATIO) * n_clusters)

    train_indices = sorted(indices[:train_end].tolist())
    validation_indices = sorted(indices[train_end:validation_end].tolist())
    test_indices = sorted(indices[validation_end:].tolist())

    output = {
        "dataset": "espargos_007_human_helmet_randomwalk_1",
        "split_level": "cluster",
        "cluster_interval_seconds": CLUSTER_INTERVAL_SECONDS,
        "random_seed": RANDOM_SEED,
        "ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO,
        },
        "n_clusters": n_clusters,
        "counts": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
        },
        "indices": {
            "train": train_indices,
            "validation": validation_indices,
            "test": test_indices,
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("=" * 80)
    print("CLUSTER SPLIT CREATED")
    print("=" * 80)
    print(f"Dataset: {output['dataset']}")
    print(f"Total clusters: {n_clusters}")
    print(f"Train: {len(train_indices)}")
    print(f"Validation: {len(validation_indices)}")
    print(f"Test: {len(test_indices)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()