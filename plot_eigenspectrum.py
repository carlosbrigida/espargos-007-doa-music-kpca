import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = "standing_center_1000.npz"

data = np.load(DATA_FILE)

csi = data["csi"]

X = csi.reshape(csi.shape[0], -1)
X = X.T

R = (X @ X.conj().T) / X.shape[1]

eigvals = np.linalg.eigvalsh(R)
eigvals = np.sort(eigvals)[::-1]

plt.figure(figsize=(10, 5))

plt.semilogy(eigvals, marker="o")

plt.grid(True)
plt.xlabel("Index")
plt.ylabel("Eigenvalue (log scale)")
plt.title("Eigenspectrum - Standing Center")

plt.tight_layout()
plt.show()