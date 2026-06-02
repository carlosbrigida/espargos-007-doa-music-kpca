import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))

ax.axis("off")

elements = [
    "ESPARGOS CSI Snapshot",
    "4 RX Arrays",
    "2 Frequency Bands",
    "4 Transmitters",
    "53 OFDM Subcarriers",
    r"$H_i \in \mathbb{C}^{4\times2\times4\times53}$"
]

y = [0.9, 0.75, 0.58, 0.41, 0.24, 0.07]

for txt, yi in zip(elements, y):
    ax.text(
        0.5,
        yi,
        txt,
        ha="center",
        va="center",
        fontsize=16,
        bbox=dict(boxstyle="round,pad=0.4")
    )

for i in range(len(y) - 1):
    ax.arrow(
        0.5,
        y[i] - 0.05,
        0,
        y[i + 1] - y[i] + 0.1,
        head_width=0.02,
        length_includes_head=True
    )

plt.tight_layout()

plt.savefig(
    "csi_structure.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()