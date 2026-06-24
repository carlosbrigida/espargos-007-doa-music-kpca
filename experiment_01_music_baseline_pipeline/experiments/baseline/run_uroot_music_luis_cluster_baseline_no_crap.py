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


def get_unitary_rootmusic_estimator(chunksize=4, shed_coeff_ratio=0):
    I = np.eye(chunksize // 2)
    J = np.flip(np.eye(chunksize // 2), axis=-1)
    Q = np.asmatrix(np.block([[I, 1.0j * I], [J, -1.0j * J]]) / np.sqrt(2))

    def unitary_rootmusic(R):
        assert len(R) == chunksize

        C = np.real(Q.H @ R @ Q)

        eig_val, eig_vec = np.linalg.eigh(C)
        eig_vec = eig_vec[:, ::-1]

        source_count = 1
        En = eig_vec[:, source_count:]

        ENSQ = Q @ En @ En.T @ Q.H

        coeffs = np.asarray(
            [np.trace(ENSQ, offset=diag) for diag in range(1, len(R))]
        )

        coeffs = coeffs[: int(len(coeffs) * (1 - shed_coeff_ratio))]
        coeffs = np.hstack((coeffs[::-1], np.trace(ENSQ), coeffs.conj()))

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

        cluster = {
            "csi_by_transmitter": csi_by_transmitter,
            "mean_position": np.mean(current_cluster["positions"], axis=0),
            "mean_timestamp": float(np.mean(current_cluster["timestamps"])),
            "datapoint_count": datapoint_count.tolist(),
        }

        clusters.append(cluster)
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
    R = np.zeros((4, 4, 4), dtype=np.complex64)

    for tx_csi in csi_by_transmitter:
        R = R + np.einsum(
            "dbrms,dbrns->bmn",
            tx_csi,
            np.conj(tx_csi),
        ) / tx_csi.shape[0]

    return R


def evaluate_scenario(scenario_name, filename):
    cache_file = PROJECT_DIR / "data" / "cache" / filename

    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}")

    data = DatasetCache(str(cache_file)).load()

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
        R_all = covariance_per_array_luis_cluster(cluster["csi_by_transmitter"])

        for array_index in range(4):
            array = RECEIVER_ARRAYS[array_index]
            true_angle_deg = true_azimuth_deg(array, cluster["mean_position"])

            electrical_angle, power = estimator(R_all[array_index])

            if np.isnan(electrical_angle):
                continue

            estimated_angle_rad = np.arcsin(
                np.clip(electrical_angle / np.pi, -1.0, 1.0)
            )

            estimated_angle_deg = np.rad2deg(estimated_angle_rad)

            error = angular_error_deg(
                estimated_angle_deg,
                true_angle_deg,
            )

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
        "mae_by_array": mae_by_array,
        "mae_global": float(np.mean(all_errors)) if len(all_errors) > 0 else np.nan,
    }


def main():
    print("=" * 80)
    print("LUIS uROOT-MUSIC BASELINE - CLUSTERS - NO CRAP")
    print("=" * 80)
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Cluster interval: {CLUSTER_INTERVAL_SECONDS} s")
    print("CRAP: disabled")
    print()

    start_time = time.perf_counter()

    results = []

    for scenario_name, filename in SCENARIOS.items():
        result = evaluate_scenario(scenario_name, filename)
        results.append(result)

        print(f"\nScenario: {scenario_name}")
        print(f"Snapshots: {result['snapshots']}")
        print(f"Clusters: {result['clusters']}")
        print(f"MAE global: {result['mae_global']:.2f}°")

        for array_index, mae in result["mae_by_array"].items():
            print(f"  Array {array_index}: {mae:.2f}°")

    elapsed = time.perf_counter() - start_time

    output = {
        "experiment": "luis_uroot_music_cluster_baseline_no_crap",
        "cluster_interval_seconds": CLUSTER_INTERVAL_SECONDS,
        "crap": False,
        "elapsed_seconds": elapsed,
        "results": results,
    }

    output_file = (
        PROJECT_DIR
        / "outputs"
        / "metrics"
        / "luis_uroot_music_cluster_baseline_no_crap.json"
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