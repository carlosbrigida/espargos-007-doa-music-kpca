from pathlib import Path
import csv

PROJECT_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_DIR / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROWS = [
    {
        "grupo": "Luiz / Baseline oficial",
        "metodo": "uRoot-MUSIC + CRAP",
        "energia": "-",
        "componentes": "-",
        "mae_a0": 3.15,
        "mae_a1": 4.67,
        "mae_a2": 6.48,
        "mae_a3": 4.33,
        "mae_global": 4.66,
        "observacao": "Resultado reproduzido exatamente da tabela do Luiz",
    },
    {
        "grupo": "Nosso experimento sem CRAP",
        "metodo": "uRoot-MUSIC",
        "energia": "-",
        "componentes": "-",
        "mae_a0": 20.57,
        "mae_a1": 55.07,
        "mae_a2": 33.20,
        "mae_a3": 42.64,
        "mae_global": 37.87,
        "observacao": "Baseline sem remoção de clutter",
    },
    {
        "grupo": "Nosso experimento sem CRAP",
        "metodo": "PCA complexa",
        "energia": "90%",
        "componentes": 4,
        "mae_a0": 20.16,
        "mae_a1": 55.08,
        "mae_a2": 33.09,
        "mae_a3": 42.69,
        "mae_global": 37.75,
        "observacao": "PCA usando representação complexa",
    },
    {
        "grupo": "Nosso experimento sem CRAP",
        "metodo": "PCA complexa",
        "energia": "95%",
        "componentes": 4,
        "mae_a0": 20.16,
        "mae_a1": 55.08,
        "mae_a2": 33.09,
        "mae_a3": 42.69,
        "mae_global": 37.75,
        "observacao": "PCA usando representação complexa",
    },
    {
        "grupo": "Nosso experimento sem CRAP",
        "metodo": "PCA complexa",
        "energia": "99%",
        "componentes": 25,
        "mae_a0": 20.63,
        "mae_a1": 55.08,
        "mae_a2": 33.21,
        "mae_a3": 42.64,
        "mae_global": 37.89,
        "observacao": "PCA usando representação complexa",
    },
    {
        "grupo": "Nosso experimento sem CRAP",
        "metodo": "PCA real/imag",
        "energia": "90%",
        "componentes": 7,
        "mae_a0": 20.17,
        "mae_a1": 54.05,
        "mae_a2": 32.98,
        "mae_a3": 42.32,
        "mae_global": 37.38,
        "observacao": "Menor MAE sem CRAP",
    },
    {
        "grupo": "Nosso experimento sem CRAP",
        "metodo": "PCA real/imag",
        "energia": "95%",
        "componentes": 8,
        "mae_a0": 20.16,
        "mae_a1": 55.08,
        "mae_a2": 33.09,
        "mae_a3": 42.69,
        "mae_global": 37.75,
        "observacao": "PCA separando real e imaginário",
    },
    {
        "grupo": "Nosso experimento sem CRAP",
        "metodo": "PCA real/imag",
        "energia": "99%",
        "componentes": 50,
        "mae_a0": 20.63,
        "mae_a1": 55.08,
        "mae_a2": 33.21,
        "mae_a3": 42.64,
        "mae_global": 37.89,
        "observacao": "PCA separando real e imaginário",
    },
    {
        "grupo": "Nosso experimento com CRAP",
        "metodo": "uRoot-MUSIC",
        "energia": "-",
        "componentes": "-",
        "mae_a0": 3.15,
        "mae_a1": 4.67,
        "mae_a2": 6.48,
        "mae_a3": 4.33,
        "mae_global": 4.66,
        "observacao": "Baseline com CRAP reproduzido",
    },
    {
        "grupo": "Nosso experimento com CRAP",
        "metodo": "PCA complexa",
        "energia": "90%",
        "componentes": 52,
        "mae_a0": 3.24,
        "mae_a1": 4.90,
        "mae_a2": 7.12,
        "mae_a3": 4.41,
        "mae_global": 4.92,
        "observacao": "Piorou em relação ao baseline com CRAP",
    },
    {
        "grupo": "Nosso experimento com CRAP",
        "metodo": "PCA complexa",
        "energia": "95%",
        "componentes": 72,
        "mae_a0": 3.20,
        "mae_a1": 4.73,
        "mae_a2": 6.63,
        "mae_a3": 4.32,
        "mae_global": 4.72,
        "observacao": "Muito próximo ao baseline com CRAP",
    },
    {
        "grupo": "Nosso experimento com CRAP",
        "metodo": "PCA complexa",
        "energia": "99%",
        "componentes": 175,
        "mae_a0": 3.15,
        "mae_a1": 4.67,
        "mae_a2": 6.49,
        "mae_a3": 4.33,
        "mae_global": 4.66,
        "observacao": "Empata com o baseline",
    },
    {
        "grupo": "Nosso experimento com CRAP",
        "metodo": "PCA real/imag",
        "energia": "90%",
        "componentes": 103,
        "mae_a0": 3.24,
        "mae_a1": 4.89,
        "mae_a2": 7.11,
        "mae_a3": 4.38,
        "mae_global": 4.91,
        "observacao": "Piorou em relação ao baseline com CRAP",
    },
    {
        "grupo": "Nosso experimento com CRAP",
        "metodo": "PCA real/imag",
        "energia": "95%",
        "componentes": 144,
        "mae_a0": 3.20,
        "mae_a1": 4.73,
        "mae_a2": 6.63,
        "mae_a3": 4.32,
        "mae_global": 4.72,
        "observacao": "Muito próximo ao baseline com CRAP",
    },
    {
        "grupo": "Nosso experimento com CRAP",
        "metodo": "PCA real/imag",
        "energia": "99%",
        "componentes": 340,
        "mae_a0": 3.15,
        "mae_a1": 4.67,
        "mae_a2": 6.49,
        "mae_a3": 4.33,
        "mae_global": 4.66,
        "observacao": "Empata com o baseline, mas usa mais componentes",
    },
]


def fmt(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_csv(rows, output_file):
    columns = list(rows[0].keys())

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, output_file):
    columns = [
        "grupo",
        "metodo",
        "energia",
        "componentes",
        "mae_a0",
        "mae_a1",
        "mae_a2",
        "mae_a3",
        "mae_global",
        "observacao",
    ]

    headers = [
        "Grupo",
        "Método",
        "Energia",
        "Componentes",
        "MAE A0",
        "MAE A1",
        "MAE A2",
        "MAE A3",
        "MAE Global",
        "Observação",
    ]

    lines = []
    lines.append("# Comparativo — Human Randomwalk\n")
    lines.append("Dataset: `espargos-0007-human-helmet-randomwalk-1.tfrecords`\n")
    lines.append("Métrica: **MAE em graus**\n")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        lines.append(
            "| "
            + " | ".join(fmt(row[column]) for column in columns)
            + " |"
        )

    lines.append("")
    lines.append("## Síntese")
    lines.append("")
    lines.append("- O baseline oficial do Luiz com CRAP foi reproduzido exatamente: MAE global = **4.66°**.")
    lines.append("- Sem CRAP, o erro sobe para **37.87°**.")
    lines.append("- A PCA sem CRAP trouxe melhora muito pequena: melhor caso = **37.38°**.")
    lines.append("- Com CRAP, a PCA não melhorou o baseline; com 99% de energia ela apenas recupera aproximadamente o mesmo desempenho.")
    lines.append("- A PCA complexa usa menos componentes que a PCA real/imag para energia equivalente.")

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main():
    csv_file = OUTPUT_DIR / "comparativo_human_randomwalk.csv"
    markdown_file = OUTPUT_DIR / "comparativo_human_randomwalk.md"

    write_csv(ROWS, csv_file)
    write_markdown(ROWS, markdown_file)

    print("=" * 80)
    print("COMPARATIVE TABLE GENERATED")
    print("=" * 80)
    print(f"CSV: {csv_file}")
    print(f"Markdown: {markdown_file}")


if __name__ == "__main__":
    main()