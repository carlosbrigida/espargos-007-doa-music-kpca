import numpy as np


def csi_to_snapshot_matrix(csi: np.ndarray) -> np.ndarray:
    if csi.ndim != 5:
        raise ValueError(f"Expected CSI with 5 dimensions, got {csi.ndim}")

    return csi.reshape(csi.shape[0], -1)


def complex_to_real_matrix(X: np.ndarray) -> np.ndarray:
    if not np.iscomplexobj(X):
        raise ValueError("Input matrix must be complex.")

    return np.concatenate([X.real, X.imag], axis=1)


def select_array(csi: np.ndarray, array_index: int = 0) -> np.ndarray:
    if csi.ndim != 5:
        raise ValueError(f"Expected CSI with 5 dimensions, got {csi.ndim}")

    return csi[:, array_index, :, :, :]


def average_subcarriers(csi_array: np.ndarray) -> np.ndarray:
    if csi_array.ndim != 4:
        raise ValueError(f"Expected array CSI with 4 dimensions, got {csi_array.ndim}")

    averaged = np.mean(csi_array, axis=-1)

    return averaged.reshape(averaged.shape[0], -1)


def covariance_per_array_luis(csi: np.ndarray) -> np.ndarray:
    """
    Luis-compatible covariance.

    Input:
        csi shape: (N, 4, 2, 4, 53)

    Output:
        R shape: (4, 4, 4)

    Uses only the 4 columns per array.
    """
    if csi.ndim != 5:
        raise ValueError(f"Expected CSI with 5 dimensions, got {csi.ndim}")

    return np.einsum(
        "dbrms,dbrns->bmn",
        csi,
        np.conj(csi),
    ) / csi.shape[0]


def covariance_per_array_8_antennas(csi: np.ndarray) -> np.ndarray:
    """
    Full 2x4 array covariance.

    Input:
        csi shape: (N, 4, 2, 4, 53)

    Output:
        R_all shape: (4, 8, 8)

    For each array:
        CSI: (N, 2, 4, 53)
        rearranged as snapshots: (N * 53, 8)
    """
    if csi.ndim != 5:
        raise ValueError(f"Expected CSI with 5 dimensions, got {csi.ndim}")

    n_arrays = csi.shape[1]
    covariance_matrices = []

    for array_index in range(n_arrays):
        csi_array = csi[:, array_index, :, :, :]

        snapshots = np.transpose(csi_array, (0, 3, 1, 2))
        snapshots = snapshots.reshape(-1, 8)

        R = snapshots.conj().T @ snapshots / snapshots.shape[0]

        covariance_matrices.append(R.astype(np.complex64))

    return np.asarray(covariance_matrices)