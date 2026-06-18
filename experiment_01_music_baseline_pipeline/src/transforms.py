import numpy as np

from sklearn.decomposition import PCA


class PCAJointTransform:
    """
    Joint PCA applied to [Re(X), Im(X)].
    """

    def __init__(self, explained_variance: float):
        if not 0.0 < explained_variance <= 1.0:
            raise ValueError("explained_variance must be in (0, 1].")

        self.explained_variance = explained_variance
        self.original_shape = None

        self.pca = PCA(
            n_components=explained_variance,
            svd_solver="full",
        )

    def fit_transform_reconstruct(self, csi: np.ndarray) -> np.ndarray:
        if csi.ndim != 5:
            raise ValueError(f"Expected CSI with 5 dimensions, got {csi.ndim}")

        if not np.iscomplexobj(csi):
            raise ValueError("CSI must be complex-valued.")

        self.original_shape = csi.shape
        n_snapshots = csi.shape[0]

        X_complex = csi.reshape(n_snapshots, -1)

        X_realimag = np.concatenate(
            [X_complex.real, X_complex.imag],
            axis=1,
        )

        Z = self.pca.fit_transform(X_realimag)
        X_reconstructed_realimag = self.pca.inverse_transform(Z)

        feature_dim = X_complex.shape[1]

        X_reconstructed_complex = (
            X_reconstructed_realimag[:, :feature_dim]
            + 1j * X_reconstructed_realimag[:, feature_dim:]
        )

        return X_reconstructed_complex.reshape(self.original_shape).astype(
            np.complex64
        )

    @property
    def n_components_(self) -> int:
        return int(self.pca.n_components_)

    @property
    def cumulative_variance_(self) -> float:
        return float(np.sum(self.pca.explained_variance_ratio_))


class PCASeparateRealImagTransform:
    """
    PCA applied separately to real and imaginary parts.

    Input:
        csi shape: (N, 4, 2, 4, 53), complex

    Procedure:
        Re(X) -> PCA -> reconstruction
        Im(X) -> PCA -> reconstruction

    Output:
        reconstructed complex CSI with original shape.
    """

    def __init__(self, explained_variance: float):
        if not 0.0 < explained_variance <= 1.0:
            raise ValueError("explained_variance must be in (0, 1].")

        self.explained_variance = explained_variance
        self.original_shape = None

        self.pca_real = PCA(
            n_components=explained_variance,
            svd_solver="full",
        )

        self.pca_imag = PCA(
            n_components=explained_variance,
            svd_solver="full",
        )

    def fit_transform_reconstruct(self, csi: np.ndarray) -> np.ndarray:
        if csi.ndim != 5:
            raise ValueError(f"Expected CSI with 5 dimensions, got {csi.ndim}")

        if not np.iscomplexobj(csi):
            raise ValueError("CSI must be complex-valued.")

        self.original_shape = csi.shape
        n_snapshots = csi.shape[0]

        X_complex = csi.reshape(n_snapshots, -1)

        X_real = X_complex.real
        X_imag = X_complex.imag

        Z_real = self.pca_real.fit_transform(X_real)
        Z_imag = self.pca_imag.fit_transform(X_imag)

        X_real_reconstructed = self.pca_real.inverse_transform(Z_real)
        X_imag_reconstructed = self.pca_imag.inverse_transform(Z_imag)

        X_reconstructed_complex = (
            X_real_reconstructed + 1j * X_imag_reconstructed
        )

        return X_reconstructed_complex.reshape(self.original_shape).astype(
            np.complex64
        )

    @property
    def n_components_real_(self) -> int:
        return int(self.pca_real.n_components_)

    @property
    def n_components_imag_(self) -> int:
        return int(self.pca_imag.n_components_)

    @property
    def cumulative_variance_real_(self) -> float:
        return float(np.sum(self.pca_real.explained_variance_ratio_))

    @property
    def cumulative_variance_imag_(self) -> float:
        return float(np.sum(self.pca_imag.explained_variance_ratio_))