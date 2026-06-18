from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path("outputs/orientation_tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# RESULTADOS MUSIC
# =========================

df = pd.DataFrame(
    {
        "Array": ["North", "West", "South", "East"],
        "Standing": [10.31, 4.45, 3.48, 43.13],
        "NW-SE": [6.07, 72.00, 3.44, 40.52],
        "SW-NE": [6.53, 73.02, 1.46, 40.19],
        "Mean": [7.64, 49.82, 2.80, 41.28],
    }
)

interpretation = pd.DataFrame(
    {
        "Array": ["North", "West", "South", "East"],
        "Assessment": [
            "Good",
            "Inconsistent",
            "Very good",
            "Consistently poor",
        ],
    }
)


def save_table(dataframe, title, filename):
    fig, ax = plt.subplots(figsize=(9, 2.5))

    ax.axis("off")

    table = ax.table(
        cellText=dataframe.values,
        colLabels=dataframe.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.7)

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    plt.tight_layout()

    output_file = OUTPUT_DIR / filename

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_file}")


save_table(
    df,
    "MUSIC Mean Absolute Error (degrees)",
    "music_results_by_array.png",
)

save_table(
    interpretation,
    "Observed Behaviour",
    "music_interpretation.png",
)

print("\nDone.")
print(f"Output directory: {OUTPUT_DIR.resolve()}")