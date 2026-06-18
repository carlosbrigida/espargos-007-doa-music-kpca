from pathlib import Path

import numpy as np


class DatasetCache:
    """
    Save and load ESPARGOS dataset cache.
    """

    def __init__(self, cache_file: str):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

    def save(self, data: dict) -> None:
        np.savez_compressed(
            self.cache_file,
            csi=data["csi"],
            mac=data["mac"],
            pos=data["pos"],
            rssi=data["rssi"],
            time=data["time"],
        )

    def load(self) -> dict:
        if not self.cache_file.exists():
            raise FileNotFoundError(f"Cache file not found: {self.cache_file}")

        data = np.load(self.cache_file, allow_pickle=True)

        return {
            "csi": data["csi"],
            "mac": data["mac"],
            "pos": data["pos"],
            "rssi": data["rssi"],
            "time": data["time"],
        }