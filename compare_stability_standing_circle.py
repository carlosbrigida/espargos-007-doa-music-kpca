import numpy as np
from scipy.linalg import subspace_angles


FILES = {
    "standing": "standing_center_1000.npz",
    "circle": "circle_1000.npz",
}

WINDOW = 100
SIGNAL_DIM = 4


def get_subspaces(file_path):
    data = np.load(file_path)
    csi = data["csi"]

    subspaces = []

    for start in range(0, csi.shape[0] - WINDOW + 1, WINDOW):
        block = csi[start:start + WINDOW]

        X = block.reshape(WINDOW, -1)
        X = X.T

        R = (X @ X.conj().T) / X.shape[1]

        eigvals, eigvecs = np.linalg.eigh(R)
        idx = np.argsort(eigvals)[::-1]

        Us = eigvecs[:, idx[:SIGNAL_DIM]]
        subspaces.append(Us)

    return subspaces


for label, file_path in FILES.items():
    print("\n==============================")
    print(label.upper())
    print("==============================")

    subspaces = get_subspaces(file_path)

    max_angles = []
    mean_angles = []

    for i in range(len(subspaces) - 1):
        angles = np.degrees(
            subspace_angles(subspaces[i], subspaces[i + 1])
        )

        max_angles.append(angles.max())
        mean_angles.append(angles.mean())

        print(
            f"{i}->{i+1}: "
            f"max={angles.max():.3f} deg | "
            f"mean={angles.mean():.3f} deg"
        )

    print("\nResumo:")
    print("Max angle médio :", np.mean(max_angles))
    print("Mean angle médio:", np.mean(mean_angles))