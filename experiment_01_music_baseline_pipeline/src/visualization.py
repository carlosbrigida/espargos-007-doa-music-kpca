from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.metrics import explained_variance_ratio


def plot_eigenvalue_characterization(
    eigvals: np.ndarray,
    output_file: str,
    max_components: int = 50,
) -> None:
    variance_ratio = explained_variance_ratio(eigvals)
    cumulative = np.cumsum(variance_ratio)

    n = min(max_components, len(eigvals))
    components = np.arange(1, n + 1)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(components, variance_ratio[:n], alpha=0.7)
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Individual explained variance")

    ax2 = ax1.twinx()
    ax2.plot(components, cumulative[:n], color="red", marker="o", linewidth=2)
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_ylim(0, 1.05)

    ax1.set_title("Eigenvalue Characterization of CSI Covariance Matrix")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def plot_method_heatmap(
    values: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    output_file: str,
) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    image = ax.imshow(values, cmap="viridis")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))

    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    ax.set_xlabel("Array")
    ax.set_ylabel("Scenario")
    ax.set_title(title)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                f"{values[i, j]:.2f}°",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Angular error (degrees)")

    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def plot_three_method_mean_comparison(
    method_labels: list[str],
    mean_errors: np.ndarray,
    output_file: str,
) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(method_labels, mean_errors, alpha=0.85)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.05,
            f"{height:.2f}°",
            ha="center",
            va="bottom",
        )

    ax.set_ylabel("Mean angular error (degrees)")
    ax.set_title("Mean Error Across Scenarios and Arrays")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def plot_array_method_comparison(
    scenario_labels: list[str],
    music_errors: np.ndarray,
    joint_errors: np.ndarray,
    separate_errors: np.ndarray,
    array_label: str,
    output_file: str,
) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(scenario_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars_music = ax.bar(
        x - width,
        music_errors,
        width,
        label="Pure MUSIC",
        alpha=0.9,
    )

    bars_joint = ax.bar(
        x,
        joint_errors,
        width,
        label="Joint PCA + MUSIC",
        alpha=0.9,
    )

    bars_separate = ax.bar(
        x + width,
        separate_errors,
        width,
        label="Separate PCA + MUSIC",
        alpha=0.9,
    )

    for bars in [bars_music, bars_joint, bars_separate]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.05,
                f"{height:.2f}°",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Angular error (degrees)")
    ax.set_title(f"Method Comparison — {array_label}")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=15, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)