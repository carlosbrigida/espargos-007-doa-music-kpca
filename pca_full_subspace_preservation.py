import numpy as np
import tensorflow as tf
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA

# ==================================================
# ESCOLHA DO DATASET
# ==================================================

# FILE = "espargos-0007-human-helmet-standing-center-1.tfrecords"

FILE = "espargos-0007-circle-1.tfrecords"

SIGNAL_DIM = 4

PCA_VARIANCES = [
    0.90,
    0.95,
    0.99,
]

# ==================================================
# CARREGAR CSI
# ==================================================

print("Carregando CSI...")
print("Arquivo:", FILE)

csi_list = []

dataset = tf.data.TFRecordDataset(FILE)

for i, raw_record in enumerate(dataset):

    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    csi = tf.io.parse_tensor(
        example.features.feature["csi"].bytes_list.value[0],
        out_type=tf.complex64,
    ).numpy()

    csi_list.append(csi)

    if (i + 1) % 5000 == 0:
        print(f"Lidos {i+1}")

csi = np.asarray(csi_list)

print("\nCSI shape:", csi.shape)

# ==================================================
# MATRIZ COMPLEXA
# ==================================================

X = csi.reshape(csi.shape[0], -1)

print("Complex shape:", X.shape)

# ==================================================
# SUBESPAÇO ORIGINAL
# ==================================================

Xh = X.T

R = (Xh @ Xh.conj().T) / Xh.shape[1]

eigvals, eigvecs = np.linalg.eigh(R)

idx = np.argsort(eigvals)[::-1]

Us_original = eigvecs[:, idx[:SIGNAL_DIM]]

# ==================================================
# PCA
# ==================================================

X_real = np.hstack(
    [
        X.real,
        X.imag,
    ]
)

print("Real shape:", X_real.shape)

for variance in PCA_VARIANCES:

    print("\n========================")
    print(f"PCA variance = {variance}")
    print("========================")

    pca = PCA(
        n_components=variance,
        svd_solver="full",
    )

    Z = pca.fit_transform(X_real)

    X_rec = pca.inverse_transform(Z)

    mse = np.mean(
        (X_real - X_rec) ** 2
    )

    n_features = X.shape[1]

    X_complex = (
        X_rec[:, :n_features]
        + 1j * X_rec[:, n_features:]
    )

    Xc = X_complex.T

    R_rec = (
        Xc @ Xc.conj().T
    ) / Xc.shape[1]

    eigvals_r, eigvecs_r = np.linalg.eigh(R_rec)

    idx_r = np.argsort(eigvals_r)[::-1]

    Us_rec = eigvecs_r[:, idx_r[:SIGNAL_DIM]]

    angles = np.degrees(
        subspace_angles(
            Us_original,
            Us_rec,
        )
    )

    print("Components:", pca.n_components_)
    print("MSE:", mse)
    print("Max angle:", angles.max())
    print("Mean angle:", angles.mean())