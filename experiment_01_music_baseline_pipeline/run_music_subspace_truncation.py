import numpy as np

from src.array_geometry import (
    steering_vector_ula,
    wavelength_from_frequency,
)
from src.cache_dataset import DatasetCache
from src.espargos_geometry import (
    RECEIVER_ARRAYS,
    true_azimuth_deg,
)
from src.metrics import angular_error_deg
from src.music import MusicEstimator
from src.preprocessing import covariance_per_array_luis


def reconstruct_covariance(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    n_components: int,
) -> np.ndarray:
    eigvals_k = eigvals[:n_components]
    eigvecs_k = eigvecs[:, :n_components]

    R_k = (
        eigvecs_k
        @ np.diag(eigvals_k)
        @ eigvecs_k.conj().T
    )

    R_k = 0.5 * (R_k + R_k.conj().T)

    return R_k


def estimate_angle(
    R: np.ndarray,
    frequency_hz: float,
    n_sources: int,
) -> tuple[float, np.ndarray]:
    music = MusicEstimator(n_sources=n_sources)

    eigvals, eigvecs = music.eigendecomposition(R)

    noise_subspace = music.noise_subspace(eigvecs)

    wavelength = wavelength_from_frequency(frequency_hz)

    angles_deg = np.linspace(-90.0, 90.0, 721)
    angles_rad = np.deg2rad(angles_deg)

    steering_vectors = np.asarray(
        [
            steering_vector_ula(
                angle_rad=angle_rad,
                n_antennas=R.shape[0],
                antenna_spacing_m=wavelength / 2.0,
                frequency_hz=frequency_hz,
            )
            for angle_rad in angles_rad
        ]
    )

    spectrum = music.pseudo_spectrum(
        noise_subspace=noise_subspace,
        steering_vectors=steering_vectors,
    )

    estimated_angle = float(
        angles_deg[np.argmax(spectrum)]
    )

    return estimated_angle, eigvals


def main() -> None:
    cache_file = (
        "data/cache/"
        "espargos_007_human_helmet_standing_center_1.npz"
    )

    array_index = 2
    n_sources = 2
    frequency_hz = 2.472e9

    cache = DatasetCache(cache_file)
    data = cache.load()

    csi = data["csi"]
    mean_position = np.mean(data["pos"], axis=0)

    array = RECEIVER_ARRAYS[array_index]

    true_angle = true_azimuth_deg(
        array,
        mean_position,
    )

    R_all = covariance_per_array_luis(csi)
    R = R_all[array_index]

    music = MusicEstimator(n_sources=n_sources)

    eigvals, eigvecs = music.eigendecomposition(R)

    print()
    print("=" * 60)
    print("ORIGINAL MUSIC DECOMPOSITION")
    print("=" * 60)

    print("Eigenvalues:")

    for i, value in enumerate(eigvals):
        print(
            f"lambda_{i+1}: "
            f"{value.real:.3f}"
        )

    print()

    print("=" * 60)
    print("SUBSPACE TRUNCATION EXPERIMENT")
    print("=" * 60)

    print(
        f"True angle: {true_angle:.2f}°"
    )

    print()

    for k in [1, 2, 3, 4]:
        R_k = reconstruct_covariance(
            eigvals=eigvals,
            eigvecs=eigvecs,
            n_components=k,
        )

        estimated_angle, _ = estimate_angle(
            R=R_k,
            frequency_hz=frequency_hz,
            n_sources=n_sources,
        )

        error = angular_error_deg(
            estimated_angle,
            true_angle,
        )

        explained = (
            np.sum(eigvals[:k])
            / np.sum(eigvals)
        )

        print(
            f"k={k} | "
            f"variance={explained:.4f} | "
            f"estimate={estimated_angle:.2f}° | "
            f"error={error:.2f}°"
        )


if __name__ == "__main__":
    main()