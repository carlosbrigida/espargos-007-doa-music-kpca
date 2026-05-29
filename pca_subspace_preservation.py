import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles

DATA_FILE = "standing_center_1000.npz"

data = np.load(DATA_FILE)

csi = data["csi"]

X = csi.reshape(csi.shape[0], -1)

# -------------------------
# Subespaço original
# -------------------------

Xh = X.T

R = (Xh @ Xh.conj().T) / Xh.shape[1]

eigvals, eigvecs = np.linalg.eigh(R)

idx = np.argsort(eigvals)[::-1]

Us_original = eigvecs[:, idx[:4]]

# -------------------------
# PCA
# -------------------------

for variance in [0.90, 0.95, 0.99]:

    print("\n========================")
    print(f"PCA variance = {variance}")
    print("========================")

    pca = PCA(n_components=variance)

    X_real = np.hstack([
        X.real,
        X.imag
    ])

    Z = pca.fit_transform(X_real)

    X_rec = pca.inverse_transform(Z)

    n_features = X.shape[1]

    X_complex = (
        X_rec[:, :n_features]
        +
        1j * X_rec[:, n_features:]
    )

    Xc = X_complex.T

    R_rec = (Xc @ Xc.conj().T) / Xc.shape[1]

    eigvals_r, eigvecs_r = np.linalg.eigh(R_rec)

    idx_r = np.argsort(eigvals_r)[::-1]

    Us_rec = eigvecs_r[:, idx_r[:4]]

    angles = np.degrees(
        subspace_angles(
            Us_original,
            Us_rec
        )
    )

    print("Components:", pca.n_components_)
    print("Max angle :", angles.max())
    print("Mean angle:", angles.mean())