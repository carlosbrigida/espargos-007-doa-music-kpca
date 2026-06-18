from pathlib import Path
import json
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "metrics"
    / "hermitian_pca_r_loso_per_array_without_crap.json"
)

BEST_N_COMPONENTS = 8


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    baseline_by_array = {i: [] for i in range(4)}
    pca_by_array = {i: [] for i in range(4)}

    for scenario_data in data["scenarios"].values():
        for result in scenario_data["baseline"]["results"]:
            baseline_by_array[result["array_index"]].append(
                result["angular_error_deg"]
            )

        selected_pca = None
        for entry in scenario_data["hermitian_pca"]:
            if entry["stats"]["n_components"] == BEST_N_COMPONENTS:
                selected_pca = entry
                break

        if selected_pca is None:
            raise RuntimeError("Could not find selected PCA entry.")

        for result in selected_pca["results"]:
            pca_by_array[result["array_index"]].append(
                result["angular_error_deg"]
            )

    print()
    print("=" * 70)
    print("HERMITIAN PCA SUMMARY BY ARRAY")
    print("=" * 70)
    print(f"Using PCA n_components = {BEST_N_COMPONENTS}")
    print()

    for array_index in range(4):
        baseline_errors = np.asarray(baseline_by_array[array_index])
        pca_errors = np.asarray(pca_by_array[array_index])

        baseline_mean = np.mean(baseline_errors)
        pca_mean = np.mean(pca_errors)
        improvement = baseline_mean - pca_mean

        print(f"Array {array_index}")
        print(f"  MUSIC mean error: {baseline_mean:.2f}°")
        print(f"  PCA mean error:   {pca_mean:.2f}°")
        print(f"  Improvement:      {improvement:.2f}°")
        print(f"  MUSIC errors:     {np.round(baseline_errors, 2)}")
        print(f"  PCA errors:       {np.round(pca_errors, 2)}")
        print()

    all_baseline = np.concatenate(
        [np.asarray(values) for values in baseline_by_array.values()]
    )
    all_pca = np.concatenate(
        [np.asarray(values) for values in pca_by_array.values()]
    )

    print("=" * 70)
    print("GLOBAL")
    print("=" * 70)
    print(f"MUSIC mean error: {np.mean(all_baseline):.2f}°")
    print(f"PCA mean error:   {np.mean(all_pca):.2f}°")
    print(f"Improvement:      {np.mean(all_baseline) - np.mean(all_pca):.2f}°")


if __name__ == "__main__":
    main()