import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FILE = "standing_center_1000.npz"

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

EIGENSPECTRUM_FIG = OUTPUT_DIR / "eigenspectrum_standing_center.png"
CUMULATIVE_FIG = OUTPUT_DIR / "cumulative_variance_standing_center.png"


def main():
    data = np.load(DATA_FILE)
    csi = data["csi"]

    print("CSI shape:", csi.shape)

    x = csi.reshape(csi.shape[0], -1)
    xh = x.T

    print("Snapshot matrix shape:", x.shape)

    covariance = (xh @ xh.conj().T) / xh.shape[1]

    eigvals = np.linalg.eigvalsh(covariance)
    eigvals = np.sort(eigvals)[::-1]

    explained_variance = eigvals / np.sum(eigvals)
    cumulative_variance = np.cumsum(explained_variance)

    n90 = np.argmax(cumulative_variance >= 0.90) + 1
    n95 = np.argmax(cumulative_variance >= 0.95) + 1
    n99 = np.argmax(cumulative_variance >= 0.99) + 1

    print("\nComponents needed:")
    print(f"90% -> {n90}")
    print(f"95% -> {n95}")
    print(f"99% -> {n99}")

    plt.figure(figsize=(8, 4))
    plt.semilogy(eigvals, linewidth=2, color="#1f77b4")
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("Component Index")
    plt.ylabel("Eigenvalue")
    plt.title("Standing Center - Eigenspectrum")
    plt.tight_layout()
    plt.savefig(EIGENSPECTRUM_FIG, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(cumulative_variance, linewidth=3, color="#1f77b4", label="Cumulative variance")
    plt.axhline(0.95, color="red", linestyle="--", linewidth=1.8, label="95% variance")
    plt.axvline(n95, color="green", linestyle="--", linewidth=1.8, label=f"{n95} PCs")
    plt.scatter(n95, cumulative_variance[n95 - 1], color="red", s=80, zorder=5)

    plt.annotate(
        f"95% variance ({n95} PCs)",
        xy=(n95, cumulative_variance[n95 - 1]),
        xytext=(8, 0.88),
        arrowprops=dict(arrowstyle="->", linewidth=1.2),
        fontsize=9,
    )

    plt.xlim(0, 50)
    plt.ylim(0.3, 1.02)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Standing Center - Cumulative Explained Variance")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(CUMULATIVE_FIG, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    print("\nFigures saved:")
    print(EIGENSPECTRUM_FIG)
    print(CUMULATIVE_FIG)


if __name__ == "__main__":
    main()