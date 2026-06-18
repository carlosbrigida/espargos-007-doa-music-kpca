from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.cache_dataset import DatasetCache
from src.clustering import cluster_snapshots


SCENARIOS = [
    "espargos_007_human_helmet_standing_center_1.npz",
    "espargos_007_human_helmet_meanders_nw_se_1.npz",
    "espargos_007_human_helmet_meanders_sw_ne_1.npz",
]

for filename in SCENARIOS:

    cache = DatasetCache(
        str(PROJECT_DIR / "data" / "cache" / filename)
    )

    data = cache.load()

    csi = data["csi"]

    clusters = cluster_snapshots(csi)

    print()
    print(filename)
    print(f"CSI shape: {csi.shape}")
    print(f"Clusters: {len(clusters)}")