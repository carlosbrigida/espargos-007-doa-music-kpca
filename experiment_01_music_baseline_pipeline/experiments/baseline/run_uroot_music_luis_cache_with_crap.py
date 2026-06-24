from pathlib import Path
import sys
import json
import time

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.cache_dataset import DatasetCache
from src.espargos_geometry import RECEIVER_ARRAYS, true_azimuth_deg
from src.metrics import angular_error_deg


SCENARIOS = {
    "standing_center": "espargos_007_human_helmet_standing_center_1.npz",
    "meanders_nw_se": "espargos_007_human_helmet_meanders_nw_se_1.npz",
    "meanders_sw_ne": "espargos_007_human_helmet_meanders_sw_ne_1.npz",
}

CLUSTER_INTERVAL_SECONDS = 1.0


def remove_clutter(csi_dataset, clutter_subspace):
    h = np.reshape(csi_dataset, (csi_dataset.shape[0], -1))
    clutter = clutter_subspace @ np.einsum(
        "sl,ds->ld",
        np.conj(clutter_subspace),
        h,
    )
    h_noclutter = h - np.transpose(clutter)
    return np.reshape(h_noclutter, csi_dataset.shape)


def get_unitary_rootmusic_estimator(chunksize=4, shed_coeff_ratio=0):
    i_matrix = np.eye(chunksize // 2)
    j_matrix = np.flip(np.eye(chunksize // 2), axis=-1)

    q_matrix = np.asmatrix(
        np.block(
            [
                [i_matrix, 1.0j * i_matrix],
                [j_matrix, -1.0j * j_matrix],
            ]
        )
        / np.sqrt(2)
    )

    def unitary_rootmusic(covariance_matrix):
        assert len(covariance_matrix) == chunksize

        transformed = np.real(q_matrix.H @ covariance_matrix @ q_matrix)

        _, eigenvectors = np.linalg.eigh(transformed)
        eigenvectors = eigenvectors[:, ::-1]

        source_count = 1
        noise_subspace = eigenvectors[:, source_count:]

        ensq = q_matrix @ noise_subspace @ noise_subspace.T @ q_matrix.H

        coeffs = np.asarray(
            [
                np.trace(ensq, offset=diag)
                for diag in range(1, len(covariance_matrix))
            ]
        )

        coeffs = coeffs[: int(len(coeffs) * (1 - shed_coeff_ratio))]
        coeffs = np.hstack((coeffs[::-1], np.trace(ensq), coeffs.conj()))

        roots = np.roots(coeffs)
        roots = roots[np.abs(roots) < 1.0]

        if roots.size == 0:
            return np.nan, np.nan

        largest_root = np.argmax(1 / (1.0 - np.abs(roots)))

        electrical_angle = np.angle(roots[largest_root])
        root_power = np.abs(roots[largest_root])

        return float(electrical_angle), float(root_power)

    return unitary_rootmusic


def cluster_cache_data(csi, mac, pos, timestamps, cluster_interval=1.0):
    unique_macs = np.unique(mac)

    clusters = []
    current_cluster = None

    def finish_cluster():
        nonlocal current_cluster

        for current_mac in unique_macs:
            current_cluster["csi_by_mac"][current_mac] = np.asarray(
                current_cluster["csi_by_mac"][current_mac]
            )

        csi_by_transmitter = [
            current_cluster["csi_by_mac"][current_mac]
            for current_mac in unique_macs
        ]

        datapoint_count = np.asarray(
            [tx_csi.shape[0] for tx_csi in csi_by_transmitter]
        )

        if np.any(datapoint_count == 0):
            current_cluster = None
            return

        clusters.append(
            {
                "csi_by_transmitter": csi_by_transmitter,
                "mean_position": np.mean(current_cluster["positions"], axis=0),
                "mean_timestamp": float(np.mean(current_cluster["timestamps"])),
                "datapoint_count": datapoint_count.tolist(),
            }
        )

        current_cluster = None

    for index in range(timestamps.shape[0]):
        if current_cluster is not None:
            if timestamps[index] > current_cluster["first_timestamp"] + cluster_interval:
                finish_cluster()

        if current_cluster is None:
            current_cluster = {
                "first_timestamp": timestamps[index],
                "positions": [],
                "timestamps": [],
                "csi_by_mac": {current_mac: [] for current_mac in unique_macs},
            }

        current_mac = mac[index]

        current_cluster["csi_by_mac"][current_mac].append(csi[index])
        current_cluster["positions"].append(pos[index])
        current_cluster["timestamps"].append(timestamps[index])

    if current_cluster is not None:
        finish_cluster()

    return clusters


def covariance_per_array_luis_cluster(csi_by_transmitter):
    covariance = np.zeros((4, 4, 4), dtype=np.complex64)

    for tx_csi in csi_by_transmitter:
        covariance = covariance + np.einsum(
            "dbrms,dbrns->bmn",
            tx_csi,
            np.conj(tx_csi),
        ) / tx_csi.shape[0]

    return covariance


def evaluate_scenario(scenario_name, filename):
    cache_file = PROJECT_DIR / "data" / "cache_luis" / filename
    clutter_file = PROJECT_DIR / "clutter_channel_estimates_luis" / f"{filename}.npy"

    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}")

    if not clutter_file.exists():
        raise FileNotFoundError(f"Clutter file not found: {clutter_file}")

    data = DatasetCache(str(cache_file)).load()
    clutter_acquisitions = np.load(clutter_file, allow_pickle=True)

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

    mae_by_array = {
        array_index: float(np.mean(errors)) if len(errors) > 0 else np.nan
        for array_index, errors in errors_by_array.items()
    }

    all_errors = [
        error
        for errors in errors_by_array.values()
        for error in errors
    ]

    return {
        "scenario": scenario_name,
        "snapshots": int(csi.shape[0]),
        "cluster_interval_seconds": CLUSTER_INTERVAL_SECONDS,
        "clusters": len(clusters),
        "crap": True,
        "cache": "cache_luis",
        "mae_by_array": mae_by_array,
        "mae_global": float(np.mean(all_errors)) if len(all_errors) > 0 else np.nan,
    }


def main():
    print("=" * 80)
    print("LUIS uROOT-MUSIC BASELINE - LUIS CACHE - WITH CRAP")
    print("=" * 80)
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Cluster interval: {CLUSTER_INTERVAL_SECONDS} s")
    print("Cache: data/cache_luis")
    print("CRAP: enabled")
    print()

    start_time = time.perf_counter()

    results = []

    for scenario_name, filename in SCENARIOS.items():
        result = evaluate_scenario(scenario_name, filename)
        results.append(result)

        print(f"\nScenario: {scenario_name}")
        print(f"Snapshots: {result['snapshots']}")
        print(f"Clusters: {result['clusters']}")
        print(f"MAE global: {result['mae_global']:.2f} deg")

        for array_index, mae in result["mae_by_array"].items():
            print(f"  Array {array_index}: {mae:.2f} deg")

    elapsed = time.perf_counter() - start_time

    output = {
        "experiment": "luis_uroot_music_luis_cache_with_crap",
        "cluster_interval_seconds": CLUSTER_INTERVAL_SECONDS,
        "cache": "data/cache_luis",
        "clutter": "clutter_channel_estimates_luis",
        "crap": True,
        "elapsed_seconds": elapsed,
        "results": results,
    }

    output_file = (
        PROJECT_DIR
        / "outputs"
        / "metrics"
        / "luis_uroot_music_luis_cache_with_crap.json"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Elapsed: {elapsed:.2f} s")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()