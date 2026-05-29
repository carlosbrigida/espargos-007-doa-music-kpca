import numpy as np


FILES = {
    "standing": "standing_center_1000.npz",
    "circle": "circle_1000.npz",
}


def analyze_file(label, file_path):
    data = np.load(file_path)
    csi = data["csi"]

    X = csi.reshape(csi.shape[0], -1)
    Xh = X.T

    R = (Xh @ Xh.conj().T) / Xh.shape[1]

    eigvals = np.linalg.eigvalsh(R)
    eigvals = np.sort(eigvals)[::-1]

    energy = eigvals / eigvals.sum()
    cum_energy = np.cumsum(energy)

    print("\n==============================")
    print(label.upper())
    print("==============================")
    print("CSI shape:", csi.shape)

    print("\nTop 10 eigenvalues:")
    for i, val in enumerate(eigvals[:10]):
        print(f"{i + 1:2d}: {val:.6e}")

    print("\nComponents needed:")
    for threshold in [0.90, 0.95, 0.99]:
        k = np.searchsorted(cum_energy, threshold) + 1
        print(f"{threshold * 100:.0f}% -> {k}")

    print("\nEigengap lambda4/lambda5:")
    print(eigvals[3] / eigvals[4])


for label, file_path in FILES.items():
    analyze_file(label, file_path)