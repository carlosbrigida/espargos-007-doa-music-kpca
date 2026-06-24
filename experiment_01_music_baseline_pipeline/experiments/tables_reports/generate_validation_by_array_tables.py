from pathlib import Path
import csv
import json

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "metrics"
    / "pca_validation_human_randomwalk.json"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = OUTPUT_DIR / "validation_human_randomwalk_by_array.csv"
MD_FILE = OUTPUT_DIR / "validation_human_randomwalk_by_array.md"
TEX_FILE = OUTPUT_DIR / "validation_human_randomwalk_by_array_abnt.tex"


METHOD_LABELS = {
    "baseline_with_crap": "uRoot-MUSIC + CRAP",
    "complex_pca_with_crap": "PCA complexa + CRAP",
    "real_imag_pca_with_crap": "PCA real/imaginária + CRAP",
}


def format_float(value):
    return f"{value:.2f}".replace(".", ",")


def load_results():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    rows = []

    for item in data["results"]:
        method = METHOD_LABELS.get(item["method"], item["method"])

        rows.append(
            {
                "metodo": method,
                "energia": (
                    "-"
                    if item["variance_threshold"] is None
                    else f"{int(item['variance_threshold'] * 100)}%"
                ),
                "componentes": (
                    "-"
                    if item["n_components"] is None
                    else str(item["n_components"])
                ),
                "compressao": f"{item['compression_ratio']:.2f}x",
                "mae_a0": item["mae_by_array"]["0"],
                "mae_a1": item["mae_by_array"]["1"],
                "mae_a2": item["mae_by_array"]["2"],
                "mae_a3": item["mae_by_array"]["3"],
                "mae_global": item["mae_global"],
            }
        )

    return rows


def write_csv(rows):
    columns = [
        "metodo",
        "energia",
        "componentes",
        "compressao",
        "mae_a0",
        "mae_a1",
        "mae_a2",
        "mae_a3",
        "mae_global",
    ]

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows):
    lines = []
    lines.append("# Validação por array — Human Randomwalk\n")
    lines.append("Dataset: `espargos-0007-human-helmet-randomwalk-1.tfrecords`\n")
    lines.append("Split: validação, 111 clusters\n")
    lines.append("Métrica: MAE em graus\n")
    lines.append("")
    lines.append(
        "| Método | Energia | Componentes | Compressão | A0 | A1 | A2 | A3 | Global |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for row in rows:
        lines.append(
            "| "
            f"{row['metodo']} | "
            f"{row['energia']} | "
            f"{row['componentes']} | "
            f"{row['compressao']} | "
            f"{format_float(row['mae_a0'])} | "
            f"{format_float(row['mae_a1'])} | "
            f"{format_float(row['mae_a2'])} | "
            f"{format_float(row['mae_a3'])} | "
            f"{format_float(row['mae_global'])} |"
        )

    with open(MD_FILE, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def write_latex(rows):
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Resultados de validação por array no cenário \textit{Human Randomwalk}}"
    )
    lines.append(r"\label{tab:validacao_human_randomwalk_por_array}")
    lines.append(r"\begin{tabular}{lcccccccc}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Método} & \textbf{Energia} & \textbf{Comp.} & "
        r"\textbf{Comp.} & \textbf{A0} & \textbf{A1} & "
        r"\textbf{A2} & \textbf{A3} & \textbf{Global} \\"
    )
    lines.append(r"\hline")

    for row in rows:
        lines.append(
            f"{row['metodo']} & "
            f"{row['energia']} & "
            f"{row['componentes']} & "
            f"{row['compressao']} & "
            f"{format_float(row['mae_a0'])} & "
            f"{format_float(row['mae_a1'])} & "
            f"{format_float(row['mae_a2'])} & "
            f"{format_float(row['mae_a3'])} & "
            f"{format_float(row['mae_global'])} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\vspace{0.2cm}")
    lines.append(r"\raggedright")
    lines.append(r"\small Fonte: Elaborado pelo autor (2026).")
    lines.append("")
    lines.append(
        r"\small Nota: Os valores correspondem ao erro absoluto médio "
        r"(MAE), em graus, calculado separadamente para cada array receptor "
        r"e também de forma global. O conjunto de validação contém 111 clusters."
    )
    lines.append(r"\end{table}")

    with open(TEX_FILE, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main():
    rows = load_results()

    write_csv(rows)
    write_markdown(rows)
    write_latex(rows)

    print("=" * 80)
    print("TABELAS POR ARRAY GERADAS")
    print("=" * 80)
    print(f"CSV: {CSV_FILE}")
    print(f"Markdown: {MD_FILE}")
    print(f"LaTeX/ABNT: {TEX_FILE}")


if __name__ == "__main__":
    main()