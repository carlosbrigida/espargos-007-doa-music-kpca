import numpy as np

from src.array_geometry import steering_vector_ula, wavelength_from_frequency
from src.cache_dataset import DatasetCache
from src.espargos_geometry import RECEIVER_ARRAYS, true_azimuth_deg
from src.metrics import angular_error_deg
from src.music import MusicEstimator
from src.preprocessing import covariance_per_array_luis


def print_step(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    cache_file = "data/cache/espargos_007_human_helmet_standing_center_1.npz"

    array_index = 2
    n_sources = 2
    frequency_hz = 2.472e9

    print_step("STEP 1 - LOAD CSI DATA")

    cache = DatasetCache(cache_file)
    data = cache.load()

    csi = data["csi"]
    positions = data["pos"]

    print(f"CSI shape: {csi.shape}")
    print("Meaning: (snapshots, arrays, rows, cols, subcarriers)")
    print(f"Number of snapshots: {csi.shape[0]}")
    print(f"Total complex coefficients per snapshot: {4 * 2 * 4 * 53}")

    print_step("STEP 2 - SELECT ARRAY")

    array = RECEIVER_ARRAYS[array_index]

    print(f"Selected array index: {array_index}")
    print(f"Selected array name: {array.name}")

    print_step("STEP 3 - BUILD MUSIC COVARIANCE MATRIX")

    R_all = covariance_per_array_luis(csi)
    R = R_all[array_index]

    print(f"R_all shape: {R_all.shape}")
    print("Meaning: (arrays, cols, cols)")
    print(f"Selected R shape: {R.shape}")
    print()
    print("This is the covariance matrix used by MUSIC.")
    print("In Luis' implementation, this matrix is 4 x 4.")

    print_step("STEP 4 - EIGENDECOMPOSITION")

    music = MusicEstimator(n_sources=n_sources)

    eigvals, eigvecs = music.eigendecomposition(R)

    print(f"Eigenvalues shape: {eigvals.shape}")
    print(f"Eigenvectors shape: {eigvecs.shape}")
    print()
    print("Eigenvalues:")
    for idx, value in enumerate(eigvals):
        print(f"lambda_{idx + 1}: {value.real:.6f}")

    print_step("STEP 5 - SIGNAL AND NOISE SUBSPACES")

    Es = music.signal_subspace(eigvecs)
    En = music.noise_subspace(eigvecs)

    print(f"n_sources: {n_sources}")
    print(f"Signal subspace Es shape: {Es.shape}")
    print(f"Noise subspace En shape: {En.shape}")
    print()
    print("MUSIC uses the noise subspace to search for directions.")
    print("The steering vector should be almost orthogonal to En at the true DoA.")

    print_step("STEP 6 - BUILD STEERING VECTORS")

    wavelength = wavelength_from_frequency(frequency_hz)
    antenna_spacing_m = wavelength / 2.0

    angles_deg = np.linspace(-90.0, 90.0, 721)
    angles_rad = np.deg2rad(angles_deg)

    steering_vectors = np.asarray(
        [
            steering_vector_ula(
                angle_rad=angle_rad,
                n_antennas=R.shape[0],
                antenna_spacing_m=antenna_spacing_m,
                frequency_hz=frequency_hz,
            )
            for angle_rad in angles_rad
        ]
    )

    print(f"Frequency: {frequency_hz / 1e9:.3f} GHz")
    print(f"Wavelength: {wavelength:.6f} m")
    print(f"Antenna spacing: {antenna_spacing_m:.6f} m")
    print(f"Angles tested: {angles_deg[0]} to {angles_deg[-1]} degrees")
    print(f"Steering vectors shape: {steering_vectors.shape}")

    print_step("STEP 7 - MUSIC PSEUDO-SPECTRUM")

    spectrum = music.pseudo_spectrum(
        noise_subspace=En,
        steering_vectors=steering_vectors,
    )

    estimated_angle = float(angles_deg[np.argmax(spectrum)])

    print(f"Spectrum shape: {spectrum.shape}")
    print(f"Estimated DoA angle: {estimated_angle:.2f} degrees")

    print_step("STEP 8 - TRUE DoA AND ERROR")

    mean_position = np.mean(positions, axis=0)
    true_angle = true_azimuth_deg(array, mean_position)
    error = angular_error_deg(estimated_angle, true_angle)

    print(f"Mean target position: {mean_position}")
    print(f"True DoA angle: {true_angle:.2f} degrees")
    print(f"Estimated DoA angle: {estimated_angle:.2f} degrees")
    print(f"Angular error: {error:.2f} degrees")

    print_step("FINAL INTERPRETATION")

    print("Where MUSIC behaves like PCA:")
    print("1. MUSIC builds the covariance matrix R.")
    print("2. MUSIC decomposes R into eigenvalues and eigenvectors.")
    print("3. The largest eigenvectors form the signal subspace.")
    print("4. The remaining eigenvectors form the noise subspace.")
    print()
    print("Where PCA/KPCA could be inserted:")
    print("A. Before R, by transforming CSI snapshots.")
    print("B. Directly on R, by modifying the eigenspace.")
    print("C. In a kernel space, replacing the linear subspace representation.")


if __name__ == "__main__":
    main()