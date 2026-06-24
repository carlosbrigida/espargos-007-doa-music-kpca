from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.cache_dataset import DatasetCache
from src.data_loader_luis import EspargosLuisTFRecordLoader


SCENARIOS = {
    "standing_center": {
        "tfrecord": (
            r"C:\Users\Carlos\Documents\Doutorado\Espargos 007\Dados"
            r"\espargos-0007-human-helmet-standing-center-1.tfrecords"
        ),
        "cache": "data/cache_luis/espargos_007_human_helmet_standing_center_1.npz",
    },
    "meanders_nw_se": {
        "tfrecord": (
            r"C:\Users\Carlos\Documents\Doutorado\Espargos 007\Dados"
            r"\espargos-0007-human-helmet-meanders-nw-se-1.tfrecords"
        ),
        "cache": "data/cache_luis/espargos_007_human_helmet_meanders_nw_se_1.npz",
    },
    "meanders_sw_ne": {
        "tfrecord": (
            r"C:\Users\Carlos\Documents\Doutorado\Espargos 007\Dados"
            r"\espargos-0007-human-helmet-meanders-sw-ne-1.tfrecords"
        ),
        "cache": "data/cache_luis/espargos_007_human_helmet_meanders_sw_ne_1.npz",
    },
}


def main():
    print("=" * 80)
    print("CREATING LUIS-COMPATIBLE CACHE")
    print("=" * 80)

    for scenario_name, cfg in SCENARIOS.items():
        print()
        print(f"Scenario: {scenario_name}")

        tfrecord_file = PROJECT_DIR / cfg["tfrecord"]
        cache_file = PROJECT_DIR / cfg["cache"]

        if not tfrecord_file.exists():
            raise FileNotFoundError(f"TFRecord not found: {tfrecord_file}")

        print(f"Loading TFRecord: {tfrecord_file}")

        loader = EspargosLuisTFRecordLoader([str(tfrecord_file)])
        data = loader.load()

        print(f"CSI shape: {data['csi'].shape}")
        print(f"MAC shape: {data['mac'].shape}")
        print(f"POS shape: {data['pos'].shape}")
        print(f"RSSI shape: {data['rssi'].shape}")
        print(f"TIME shape: {data['time'].shape}")

        cache = DatasetCache(str(cache_file))
        cache.save(data)

        print(f"Saved: {cache_file}")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()