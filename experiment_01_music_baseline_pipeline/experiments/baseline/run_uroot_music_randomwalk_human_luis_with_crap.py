from src.run_uroot_music_luis_cache_with_crap import (
    PROJECT_DIR,
    CLUSTER_INTERVAL_SECONDS,
    remove_clutter,
    get_unitary_rootmusic_estimator,
    cluster_cache_data,
    covariance_per_array_luis_cluster,
)

import json
import time
import numpy as np

from src.cache_dataset import DatasetCache
from src.espargos_geometry import RECEIVER_ARRAYS, true_azimuth_deg
from src.metrics import angular_error_deg


CACHE_FILE = (
    PROJECT_DIR
    / "data"
    / "cache_luis"
    / "espargos_007_human_helmet_randomwalk_1.npz"
)

CLUTTER_FILE = (
    PROJECT_DIR
    / "clutter_channel_estimates_luis"
    / "espargos_007_human_helmet_randomwalk_1.npz.npy"
)


def main():
    print("=" * 80)
    print("uROOT-MUSIC - HUMAN RANDOMWALK - LUIS CACHE - WITH CRAP")
    print("=" * 80)

    data = DatasetCache(str(CACHE_FILE)).load()
    clutter_acquisitions = np.load(CLUTTER_FILE, allow_pickle=True)

    csi = data["csi"]
    mac = data["mac"]
    pos = data["pos"]
    timestamps = data["time"]

    clusters = cluster_cache_data(
        csi=csi,
        mac=mac,
        pos=pos,
        timestamps=timestamps,
        cluster_interval=CLUSTER_INTERVAL_SECONDS,
    )

    estimator = get_unitary_rootmusic_estimator(4)
    errors_by_array = {array_index: [] for array_index in range(4)}

    start_time = time.perf_counter()

    for cluster in clusters:
        csi_by_transmitter_noclutter = []

        for tx_index, tx_csi in enumerate(cluster["csi_by_transmitter"]):
            tx_csi_noclutter = remove_clutter(
                tx_csi,
                clutter_acquisitions[tx_index],
            )
            csi_by_transmitter_noclutter.append(tx_csi_noclutter)

        covariance_all = covariance_per_array_luis_cluster(
            csi_by_transmitter_noclutter
        )

        for array_index in range(4):
            array = RECEIVER_ARRAYS[array_index]
            true_angle_deg = true_azimuth_deg(array, cluster["mean_position"])

            electrical_angle, _ = estimator(covariance_all[array_index])

            if np.isnan(electrical_angle):
                continue

            estimated_angle_rad = np.arcsin(
                np.clip(electrical_angle / np.pi, -1.0, 1.0)
            )

            estimated_angle_deg = np.rad2deg(estimated_angle_rad)
            error = angular_error_deg(estimated_angle_deg, true_angle_deg)

            errors_by_array[array_index].append(error)

    elapsed = time.perf_counter() - start_time

    mae_by_array = {
        array_index: float(np.mean(errors))
        for array_index, errors in errors_by_array.items()
    }

    all_errors = [
        error
        for errors in errors_by_array.values()
        for error in errors
    ]

    result = {
        "experiment": "uroot_music_randomwalk_human_luis_with_crap",
        "cache": str(CACHE_FILE),
        "clutter": str(CLUTTER_FILE),
        "snapshots": int(csi.shape[0]),
        "clusters": len(clusters),
        "cluster_interval_seconds": CLUSTER_INTERVAL_SECONDS,
        "crap": True,
        "mae_by_array": mae_by_array,
        "mae_global": float(np.mean(all_errors)),
        "elapsed_seconds": elapsed,
    }

    output_file = (
        PROJECT_DIR
        / "outputs"
        / "metrics"
        / "uroot_music_randomwalk_human_luis_with_crap.json"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print(f"Snapshots: {result['snapshots']}")
    print(f"Clusters: {result['clusters']}")
    print(f"MAE global: {result['mae_global']:.2f} deg")

    for array_index, mae in mae_by_array.items():
        print(f"Array {array_index}: {mae:.2f} deg")

    print(f"Saved to: {output_file}")
    print(f"Elapsed: {elapsed:.2f} s")


if __name__ == "__main__":
    main()