import numpy as np

DATA_FILE = "standing_center_1000.npz"

data = np.load(DATA_FILE)

csi = data["csi"]

print("CSI shape:", csi.shape)

# --------------------------------------------------
# Transformar cada snapshot em um vetor complexo
# --------------------------------------------------

n_snapshots = csi.shape[0]

X = csi.reshape(n_snapshots, -1)

print("X shape:", X.shape)

# --------------------------------------------------
# Covariância
# --------------------------------------------------

X = X.T  # (features, snapshots)

R = (X @ X.conj().T) / X.shape[1]

print("Covariance shape:", R.shape)

# --------------------------------------------------
# Autovalores
# --------------------------------------------------

eigvals = np.linalg.eigvalsh(R)

eigvals = np.sort(eigvals)[::-1]

print("\nTop 20 eigenvalues:\n")

for i, val in enumerate(eigvals[:20]):
    print(f"{i+1:2d}: {val:.6e}")

# --------------------------------------------------
# Energia acumulada
# --------------------------------------------------

energy = eigvals / eigvals.sum()
cum_energy = np.cumsum(energy)

print("\nComponents needed:")

for threshold in [0.90, 0.95, 0.99]:
    k = np.searchsorted(cum_energy, threshold) + 1
    print(f"{threshold*100:.0f}% -> {k}")