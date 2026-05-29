import numpy as np
from sklearn.decomposition import PCA, KernelPCA

DATA_FILE = "standing_center_1000.npz"

data = np.load(DATA_FILE)

csi = data["csi"]

X = csi.reshape(csi.shape[0], -1)

X_real = np.hstack([
    X.real,
    X.imag
])

print("Shape:", X_real.shape)

# --------------------------------------------------
# PCA
# --------------------------------------------------

print("\n===== PCA =====")

for variance in [0.90, 0.95, 0.99]:

    pca = PCA(n_components=variance)

    Z = pca.fit_transform(X_real)

    X_rec = pca.inverse_transform(Z)

    mse = np.mean((X_real - X_rec) ** 2)

    print(
        f"variance={variance} | "
        f"components={pca.n_components_} | "
        f"MSE={mse:.6f}"
    )

# --------------------------------------------------
# KPCA
# --------------------------------------------------

print("\n===== KPCA =====")

for comps in [4, 8, 16]:

    kpca = KernelPCA(
        n_components=comps,
        kernel="rbf",
        gamma=1e-5,
        fit_inverse_transform=True,
    )

    Z = kpca.fit_transform(X_real)

    X_rec = kpca.inverse_transform(Z)

    mse = np.mean((X_real - X_rec) ** 2)

    print(
        f"components={comps} | "
        f"MSE={mse:.6f}"
    )