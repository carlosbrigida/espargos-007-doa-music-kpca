from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

CACHE_FILE = (
    PROJECT_DIR
    / "data"
    / "cache_luis"
    / "espargos_007_human_helmet_randomwalk_1.npz"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "clutter_channel_estimates_luis"
    / "espargos_007_human_helmet_randomwalk_1.npz.npy"
)

CRAP_ORDER = 2


def acquire_clutter(csi_dataset, order=2):
    c = np.reshape(csi_dataset, (csi_dataset.shape[0], -1))
    r = np.einsum("na,nb->ab", c, np.conj(c), optimize=True)
    _, v = np.linalg.eigh(r)
    return v[:, ::-1][:, :order]


def main():
    print("=" * 80)
    print("GENERATING CLUTTER - HUMAN RANDOMWALK - LUIS CACHE")
    print("=" * 80)

    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"Cache not found: {CACHE_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(CACHE_FILE, allow_pickle=True)

    csi = data["csi"]
    mac = data["mac"]

    unique_macs = np.unique(mac)

    clutter_by_tx = []

    print(f"Cache: {CACHE_FILE}")
    print(f"CSI shape: {csi.shape}")
    print(f"CRAP order: {CRAP_ORDER}")
    print()

    for tx_index, current_mac in enumerate(unique_macs):
        tx_csi = csi[mac == current_mac]

        print(f"TX {tx_index}: snapshots={tx_csi.shape[0]}")

        clutter = acquire_clutter(tx_csi, order=CRAP_ORDER)
        clutter_by_tx.append(clutter)

        print(f"  clutter shape: {clutter.shape}")

    clutter_by_tx = np.asarray(clutter_by_tx)

    np.save(OUTPUT_FILE, clutter_by_tx)

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()