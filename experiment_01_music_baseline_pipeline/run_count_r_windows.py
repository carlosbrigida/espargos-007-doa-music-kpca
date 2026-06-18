from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.cache_dataset import DatasetCache


SCENARIOS = {
    "standing_center": "espargos_007_human_helmet_standing_center_1.npz",
    "meanders_nw_se": "espargos_007_human_helmet_meanders_nw_se_1.npz",
    "meanders_sw_ne": "espargos_007_human_helmet_meanders_sw_ne_1.npz",
}


def count_windows(n_snapshots: int, window_size: int, stride: int) -> int:
    if n_snapshots < window_size:
        return 0

    return 1 + (n_snapshots - window_size) // stride


def main() -> None:
    window_sizes = [50, 100, 200, 500]
    strides = [25, 50, 100, 250]

    print()
    print("=" * 70)
    print("COUNT TEMPORAL WINDOWS FOR R TRAINING")
    print("=" * 70)

    for scenario_name, filename in SCENARIOS.items():
        cache_file = PROJECT_DIR / "data" / "cache" / filename
        data = DatasetCache(str(cache_file)).load()

        csi = data["csi"]
        n_snapshots = csi.shape[0]

        print()
        print(f"Scenario: {scenario_name}")
        print(f"CSI shape: {csi.shape}")
        print(f"Snapshots: {n_snapshots}")

        for window_size in window_sizes:
            for stride in strides:
                n_windows = count_windows(
                    n_snapshots=n_snapshots,
                    window_size=window_size,
                    stride=stride,
                )

                n_r_matrices = n_windows * 4

                print(
                    f"  window={window_size:>3}, stride={stride:>3} "
                    f"-> windows={n_windows:>4}, R matrices={n_r_matrices:>5}"
                )


if __name__ == "__main__":
    main()