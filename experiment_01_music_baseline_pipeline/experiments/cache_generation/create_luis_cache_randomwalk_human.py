from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.cache_dataset import DatasetCache
from src.data_loader_luis import EspargosLuisTFRecordLoader


TFRECORD_FILE = (
    r"C:\Users\Carlos\Documents\Doutorado\Espargos 007\Dados"
    r"\espargos-0007-human-helmet-randomwalk-1.tfrecords"
)

CACHE_FILE = (
    PROJECT_DIR
    / "data"
    / "cache_luis"
    / "espargos_007_human_helmet_randomwalk_1.npz"
)


def main():
    print("=" * 80)
    print("CREATING LUIS-COMPATIBLE CACHE - HUMAN RANDOMWALK")
    print("=" * 80)

    tfrecord_file = Path(TFRECORD_FILE)

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

    cache = DatasetCache(str(CACHE_FILE))
    cache.save(data)

    print(f"Saved: {CACHE_FILE}")

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()