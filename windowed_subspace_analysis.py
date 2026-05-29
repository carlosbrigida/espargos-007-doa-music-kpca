import numpy as np

DATA_FILE = "standing_center_1000.npz"

WINDOW = 100

data = np.load(DATA_FILE)

csi = data["csi"]

n_snapshots = csi.shape[0]

print("CSI shape:", csi.shape)

for start in range(0, n_snapshots - WINDOW + 1, WINDOW):

    stop = start + WINDOW

    block = csi[start:stop]

    X = block.reshape(WINDOW, -1)
    X = X.T

    R = (X @ X.conj().T) / X.shape[1]

    eigvals = np.linalg.eigvalsh(R)
    eigvals = np.sort(eigvals)[::-1]

    print(
        f"Window {start:4d}-{stop:4d} | "
        f"{eigvals[0]:.2e} "
        f"{eigvals[1]:.2e} "
        f"{eigvals[2]:.2e} "
        f"{eigvals[3]:.2e}"
    )