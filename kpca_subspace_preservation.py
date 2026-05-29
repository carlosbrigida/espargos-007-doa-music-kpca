import numpy as np
from scipy.linalg import subspace_angles
from sklearn.decomposition import KernelPCA

print("INICIO")

data = np.load("standing_center_1000.npz")
csi = data["csi"]

print("Dados carregados")

X = csi.reshape(csi.shape[0], -1)
X_real = np.hstack([X.real, X.imag])

print("Shape:", X_real.shape)

kpca = KernelPCA(
    n_components=8,
    kernel="rbf",
    gamma=1e-6,
    fit_inverse_transform=True,
)

print("Antes fit")

Z = kpca.fit_transform(X_real)

print("Depois fit")

X_rec = kpca.inverse_transform(Z)

print("Depois inverse")

mse = np.mean((X_real - X_rec) ** 2)

print("MSE:", mse)

print("FIM")