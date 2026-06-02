import matplotlib.pyplot as plt

OUTPUT_FILE = "figures/snapshot_pipeline.png"

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis("off")

steps = [
    r"CSI Tensor" + "\n" + r"$H_i \in \mathbb{C}^{4\times2\times4\times53}$",
    r"Vectorization" + "\n" + r"$x_i = \mathrm{vec}(H_i)$",
    r"Complex Snapshot Vector" + "\n" + r"$x_i \in \mathbb{C}^{1696}$",
    r"Stack N Snapshots" + "\n" + r"$X \in \mathbb{C}^{N\times1696}$",
    r"Real + Imag Concatenation" + "\n" + r"$X_r \in \mathbb{R}^{N\times3392}$"
]

y = [0.90, 0.72, 0.54, 0.36, 0.18]

for txt, yi in zip(steps, y):
    ax.text(
        0.5,
        yi,
        txt,
        ha="center",
        va="center",
        fontsize=16,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#EAF3FA",
            edgecolor="#1F5A8A",
            linewidth=2
        )
    )

for i in range(len(y)-1):
    ax.arrow(
        0.5,
        y[i]-0.06,
        0,
        y[i+1]-y[i]+0.12,
        head_width=0.015,
        head_length=0.02,
        linewidth=2,
        color="#1F5A8A",
        length_includes_head=True
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()