import numpy as np
from scipy.linalg import subspace_angles

DATA_FILE = "standing_center_1000.npz"

WINDOW = 100
SIGNAL_DIM = 4

data = np.load(DATA_FILE)

csi = data["csi"]

subspaces = []

for start in range(0, csi.shape[0] - WINDOW + 1, WINDOW):

    block = csi[start:start + WINDOW]

    X = block.reshape(WINDOW, -1)
    X = X.T

    R = (X @ X.conj().T) / X.shape[1]

    eigvals, eigvecs = np.linalg.eigh(R)

    idx = np.argsort(eigvals)[::-1]

    eigvecs = eigvecs[:, idx]

    Us = eigvecs[:, :SIGNAL_DIM]

    subspaces.append(Us)

print("\nPrincipal angles between consecutive windows\n")

for i in range(len(subspaces) - 1):

    angles = subspace_angles(
        subspaces[i],
        subspaces[i + 1]
    )

    angles_deg = np.degrees(angles)

    print(
        f"{i}->{i+1}: "
        f"max={angles_deg.max():.3f} deg | "
        f"mean={angles_deg.mean():.3f} deg"
    )