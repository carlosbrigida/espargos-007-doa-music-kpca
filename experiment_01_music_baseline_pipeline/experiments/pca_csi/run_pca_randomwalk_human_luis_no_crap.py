from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.cache_dataset import DatasetCache
from src.espargos_geometry import RECEIVER_ARRAYS, true_azimuth_deg
from src.metrics import angular_error_deg
from src.run_uroot_music_luis_cache_with_crap import (
    cluster_cache_data,
    covariance_per_array_luis_cluster,
    get_unitary_rootmusic_estimator,
)

CACHE_FILE = (
    PROJECT_DIR
    / "data"
    / "cache_luis"
    / "espargos_007_human_helmet_randomwalk_1.npz"
)

CLUSTER_INTERVAL_SECONDS = 1.0
VARIANCES = [0.90, 0.95, 0.99]

def flatten_csi(csi):
    return csi.reshape(csi.shape[0], -1)


def unflatten_csi(x_flat, original_shape):
    return x_flat.reshape(original_shape)


def complex_pca_reconstruct(csi, variance_threshold):
    """
    PCA complexa via SVD.

    Mantém os números complexos juntos.
    A energia é calculada usando os valores singulares.
    """
    original_shape = csi.shape
    x = flatten_csi(csi)

    mean = np.mean(x, axis=0, keepdims=True)
    x_centered = x - mean

    u, singular_values, vh = np.linalg.svd(
        x_centered,
        full_matrices=False,
    )

    energy = singular_values ** 2
    cumulative_energy = np.cumsum(energy) / np.sum(energy)

    n_components = int(np.searchsorted(cumulative_energy, variance_threshold) + 1)

    x_reconstructed = (
        u[:, :n_components]
        @ np.diag(singular_values[:n_components])
        @ vh[:n_components, :]
    )

    x_reconstructed = x_reconstructed + mean

    preserved_energy = float(cumulative_energy[n_components - 1])

    return unflatten_csi(x_reconstructed, original_shape), n_components, preserved_energy


def real_imag_pca_reconstruct(csi, variance_threshold):
    """
    PCA real separando parte real e imaginária.

    X complexo vira [Re(X), Im(X)].
    Depois reconstruímos X = Re + j Im.
    """
    original_shape = csi.shape
    x = flatten_csi(csi)

    x_realimag = np.concatenate([x.real, x.imag], axis=1)

    pca = PCA(n_components=variance_threshold, svd_solver="full")
    z = pca.fit_transform(x_realimag)
    x_reconstructed_realimag = pca.inverse_transform(z)

    n_features = x.shape[1]

    real_part = x_reconstructed_realimag[:, :n_features]
    imag_part = x_reconstructed_realimag[:, n_features:]

    x_reconstructed = real_part + 1j * imag_part

    n_components = int(pca.n_components_)
    preserved_energy = float(np.sum(pca.explained_variance_ratio_))

    return (
        unflatten_csi(x_reconstructed.astype(np.complex64), original_shape),
        n_components,
        preserved_energy,
    )
def evaluate_csi(csi, mac, pos, timestamps):
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
        covariance_all = covariance_per_array_luis_cluster(
            cluster["csi_by_transmitter"]
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
        array_index: float(np.mean(errors))
        for array_index, errors in errors_by_array.items()
    }

    all_errors = [
        error
        for errors in errors_by_array.values()
        for error in errors
    ]

    return {
        "clusters": len(clusters),
        "mae_by_array": mae_by_array,
        "mae_global": float(np.mean(all_errors)),
    }
def main():
    print("=" * 80)
    print("PCA EXPERIMENT - HUMAN RANDOMWALK - NO CRAP")
    print("=" * 80)

    data = DatasetCache(str(CACHE_FILE)).load()

    csi = data["csi"]
    mac = data["mac"]
    pos = data["pos"]
    timestamps = data["time"]

    print(f"CSI shape: {csi.shape}")
    print()

    print("=" * 80)
    print("BASELINE")
    print("=" * 80)

    baseline = evaluate_csi(
        csi=csi,
        mac=mac,
        pos=pos,
        timestamps=timestamps,
    )

    print(f"Clusters: {baseline['clusters']}")
    print(f"MAE global: {baseline['mae_global']:.2f} deg")

    for array_index, mae in baseline["mae_by_array"].items():
        print(f"Array {array_index}: {mae:.2f} deg")

    print()

    for variance in VARIANCES:

        print("=" * 80)
        print(f"COMPLEX PCA - ENERGY {variance:.2f}")
        print("=" * 80)

        csi_pca, n_components, preserved = complex_pca_reconstruct(
            csi,
            variance,
        )

        result = evaluate_csi(
            csi=csi_pca,
            mac=mac,
            pos=pos,
            timestamps=timestamps,
        )

        print(f"Components: {n_components}")
        print(f"Preserved energy: {preserved:.6f}")
        print(f"MAE global: {result['mae_global']:.2f} deg")

        for array_index, mae in result["mae_by_array"].items():
            print(f"Array {array_index}: {mae:.2f} deg")

        print()

    for variance in VARIANCES:

        print("=" * 80)
        print(f"REAL/IMAG PCA - ENERGY {variance:.2f}")
        print("=" * 80)

        csi_pca, n_components, preserved = real_imag_pca_reconstruct(
            csi,
            variance,
        )

        result = evaluate_csi(
            csi=csi_pca,
            mac=mac,
            pos=pos,
            timestamps=timestamps,
        )

        print(f"Components: {n_components}")
        print(f"Preserved energy: {preserved:.6f}")
        print(f"MAE global: {result['mae_global']:.2f} deg")

        for array_index, mae in result["mae_by_array"].items():
            print(f"Array {array_index}: {mae:.2f} deg")

        print()

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()