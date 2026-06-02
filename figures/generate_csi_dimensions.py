import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,5))

ax.axis("off")

labels = [
    "4 RX Arrays",
    "2 Bands",
    "4 TX",
    "53 OFDM Subcarriers"
]

x = [0.1,0.35,0.6,0.85]

for xi, label in zip(x, labels):
    ax.text(
        xi,
        0.6,
        label,
        ha="center",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.5")
    )

for i in range(3):
    ax.arrow(
        x[i]+0.08,
        0.6,
        x[i+1]-x[i]-0.16,
        0,
        head_width=0.03,
        length_includes_head=True
    )

ax.text(
    0.5,
    0.2,
    r"$H_i \in \mathbb{C}^{4\times2\times4\times53}$",
    fontsize=18,
    ha="center"
)

plt.tight_layout()

plt.savefig(
    "figures/csi_dimensions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()