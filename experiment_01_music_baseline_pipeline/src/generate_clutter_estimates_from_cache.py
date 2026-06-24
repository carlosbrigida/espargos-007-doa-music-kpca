from pathlib import Path
import sys
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


SCENARIOS = {
    "standing_center": "espargos_007_human_helmet_standing_center_1.npz",
    "meanders_nw_se": "espargos_007_human_helmet_meanders_nw_se_1.npz",
    "meanders_sw_ne": "espargos_007_human_helmet_meanders_sw_ne_1.npz",
}

CRAP_ORDER = 2


def acquire_clutter(csi_dataset, order=2):
    c = np.reshape(csi_dataset, (csi_dataset.shape[0], -1))
    r = np.einsum("na,nb->ab", c, np.conj(c), optimize=True)
    _, v = np.linalg.eigh(r)
    return v[:, ::-1][:, :order]


def main():
    cache_dir = PROJECT_DIR / "data" / "cache"
    output_dir = PROJECT_DIR / "clutter_channel_estimates"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING CRAP CLUTTER ESTIMATES FROM CACHE")
    print("=" * 80)
    print(f"Project dir: {PROJECT_DIR}")
    print(f"CRAP order: {CRAP_ORDER}")
    print(f"Output dir: {output_dir}")
    print()

    for scenario_name, filename in SCENARIOS.items():
        cache_file = cache_dir / filename

        if not cache_file.exists():
            raise FileNotFoundError(f"Cache not found: {cache_file}")

        print(f"Scenario: {scenario_name}")
        print(f"Loading: {cache_file}")

        data = np.load(cache_file, allow_pickle=True)
        csi = data["csi"]
        mac = data["mac"]

        unique_macs = np.unique(mac)

        clutter_by_tx = []

        for tx_index, current_mac in enumerate(unique_macs):
            print(f"  TX {tx_index}: {current_mac}")

            tx_csi = csi[mac == current_mac]

            if tx_csi.shape[0] == 0:
                raise ValueError(f"No CSI found for MAC: {current_mac}")

            clutter = acquire_clutter(tx_csi, order=CRAP_ORDER)
            clutter_by_tx.append(clutter)

            print(f"    CSI shape: {tx_csi.shape}")
            print(f"    Clutter shape: {clutter.shape}")

        clutter_by_tx = np.asarray(clutter_by_tx)

        output_file = output_dir / f"{filename}.npy"
        np.save(output_file, clutter_by_tx)

        print(f"Saved: {output_file}")
        print()

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()