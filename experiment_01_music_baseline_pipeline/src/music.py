import numpy as np


class MusicEstimator:
    """
    MUSIC estimator with optional PCA-based covariance reconstruction.
    """

    def __init__(self, n_sources: int):
        if n_sources < 1:
            raise ValueError("n_sources must be >= 1")

        self.n_sources = n_sources

    def covariance(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError(f"Expected X with 2 dimensions, got {X.ndim}")

        return (X.conj().T @ X) / X.shape[0]

    def eigendecomposition(self, R: np.ndarray):
        eigvals, eigvecs = np.linalg.eigh(R)

        order = np.argsort(eigvals)[::-1]

        return eigvals[order], eigvecs[:, order]

    def signal_subspace(self, eigvecs: np.ndarray) -> np.ndarray:
        return eigvecs[:, : self.n_sources]

    def noise_subspace(self, eigvecs: np.ndarray) -> np.ndarray:
        return eigvecs[:, self.n_sources :]

    def reconstruct_covariance_by_variance(
        self,
        R: np.ndarray,
        variance_threshold: float,
    ) -> tuple[np.ndarray, int, float]:
        """
        PCA-like reconstruction inside MUSIC.

        The covariance matrix R is decomposed and reconstructed
        using the smallest number of eigencomponents required
        to preserve the desired variance.
        """
        if not 0.0 < variance_threshold <= 1.0:
            raise ValueError("variance_threshold must be in (0, 1].")

        eigvals, eigvecs = self.eigendecomposition(R)

        eigvals = np.real(eigvals)

        total_energy = np.sum(eigvals)

        if total_energy <= 0:
            raise ValueError("Covariance eigenvalue energy must be positive.")

        cumulative_energy = np.cumsum(eigvals) / total_energy

        n_components = int(
            np.searchsorted(cumulative_energy, variance_threshold) + 1
        )

        selected_eigvals = eigvals[:n_components]
        selected_eigvecs = eigvecs[:, :n_components]

        R_reconstructed = (
            selected_eigvecs
            @ np.diag(selected_eigvals)
            @ selected_eigvecs.conj().T
        )

        R_reconstructed = 0.5 * (
            R_reconstructed + R_reconstructed.conj().T
        )

        preserved_variance = float(cumulative_energy[n_components - 1])

        return R_reconstructed, n_components, preserved_variance

    def pseudo_spectrum(
        self,
        noise_subspace: np.ndarray,
        steering_vectors: np.ndarray,
    ) -> np.ndarray:
        projection = noise_subspace @ noise_subspace.conj().T

        spectrum = []

        for steering_vector in steering_vectors:
            denominator = (
                steering_vector.conj().T
                @ projection
                @ steering_vector
            )

            value = 1.0 / np.abs(denominator.squeeze())
            spectrum.append(value)

        return np.asarray(spectrum)