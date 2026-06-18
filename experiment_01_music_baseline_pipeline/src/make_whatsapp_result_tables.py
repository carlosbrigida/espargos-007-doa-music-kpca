from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("outputs/whatsapp_tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_table_image(df: pd.DataFrame, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    ax.set_title(title, fontsize=13, weight="bold", pad=14)

    output_path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def save_markdown(df: pd.DataFrame, title: str, filename: str) -> None:
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(f"{title}\n\n")
        file.write(df.to_markdown(index=False))

    print(f"Saved: {output_path}")


def main() -> None:
    # Tabela do TCC do Luís - cenário humano
    tcc_df = pd.DataFrame(
        {
            "Algoritmo": ["MUSIC", "Unitary Root-MUSIC"],
            "Array 0": ["37.49°", "3.15°"],
            "Array 1": ["48.95°", "4.67°"],
            "Array 2": ["50.92°", "6.48°"],
            "Array 3": ["35.91°", "4.33°"],
        }
    )

    # Resultados atuais obtidos nos nossos experimentos com MUSIC clássico
    current_df = pd.DataFrame(
        {
            "Algoritmo": ["MUSIC atual"],
            "Array 0": ["7.64°"],
            "Array 1": ["49.82°"],
            "Array 2": ["2.79°"],
            "Array 3": ["41.28°"],
        }
    )

    # Comparação direta apenas com MUSIC clássico
    comparison_df = pd.DataFrame(
        {
            "Array": ["Array 0", "Array 1", "Array 2", "Array 3"],
            "MUSIC TCC Luís": ["37.49°", "48.95°", "50.92°", "35.91°"],
            "MUSIC atual": ["7.64°", "49.82°", "2.79°", "41.28°"],
            "Observação": [
                "Atual menor",
                "Próximo",
                "Atual menor",
                "Próximo",
            ],
        }
    )

    # Resultados atuais por cenário
    scenario_df = pd.DataFrame(
        {
            "Array": ["Array 0", "Array 1", "Array 2", "Array 3"],
            "Standing": ["10.31°", "4.45°", "3.48°", "43.13°"],
            "NW-SE": ["6.07°", "72.00°", "3.44°", "40.52°"],
            "SW-NE": ["6.53°", "73.02°", "1.46°", "40.19°"],
            "Média": ["7.64°", "49.82°", "2.79°", "41.28°"],
        }
    )

    tables = [
        (
            tcc_df,
            "TCC Luís - MAE Humano por Array",
            "01_tcc_luis_mae_humano.png",
            "01_tcc_luis_mae_humano.md",
        ),
        (
            current_df,
            "Experimento Atual - MUSIC Clássico",
            "02_music_atual.png",
            "02_music_atual.md",
        ),
        (
            comparison_df,
            "Comparação MUSIC: TCC Luís vs Atual",
            "03_comparacao_music.png",
            "03_comparacao_music.md",
        ),
        (
            scenario_df,
            "Experimento Atual por Cenário",
            "04_music_atual_por_cenario.png",
            "04_music_atual_por_cenario.md",
        ),
    ]

    for df, title, image_name, md_name in tables:
        save_table_image(df, title, image_name)
        save_markdown(df, title, md_name)

    print("\nArquivos gerados em:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()