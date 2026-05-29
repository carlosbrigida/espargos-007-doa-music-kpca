import numpy as np
import matplotlib.pyplot as plt

data = np.load("circle_1000.npz")

pos = data["pos"]

plt.figure(figsize=(8, 8))

sc = plt.scatter(
    pos[:, 0],
    pos[:, 1],
    c=np.arange(len(pos)),
    s=8,
    cmap="viridis"
)

plt.colorbar(sc, label="Snapshot index")

plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Circle trajectory - first 1000 snapshots")

plt.axis("equal")
plt.grid(True)

plt.tight_layout()
plt.show()