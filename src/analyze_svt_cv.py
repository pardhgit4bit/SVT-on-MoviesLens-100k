
# src/analyze_svt_cv.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TAU_MULTIPLIERS = [8, 9, 10]
N_FOLDS = 5


def load_histories():

    records = []

    for multiplier in TAU_MULTIPLIERS:

        for fold in range(1, N_FOLDS + 1):

            path = (
                TABLES_DIR /
                f"svt_tau_{multiplier}_fold_{fold}_history.csv"
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing history file: {path}"
                )

            history = pd.read_csv(path)

            history["tau_multiplier"] = multiplier
            history["fold"] = fold

            records.append(history)

    return pd.concat(records, ignore_index=True)


def aggregate_curves(data):

    curves = (
        data.groupby(
            ["tau_multiplier", "iteration"]
        )
        .agg(
            validation_rmse_mean=(
                "validation_rmse", "mean"
            ),
            validation_rmse_std=(
                "validation_rmse", "std"
            ),
            validation_mae_mean=(
                "validation_mae", "mean"
            ),
            rank_mean=(
                "rank", "mean"
            ),
            observed_error_mean=(
                "relative_observed_error", "mean"
            )
        )
        .reset_index()
    )

    curves.to_csv(
        TABLES_DIR / "svt_cv_iteration_summary.csv",
        index=False
    )

    return curves


def find_best_common_config(curves):

    best_rows = []

    for multiplier in TAU_MULTIPLIERS:

        subset = curves[
            curves["tau_multiplier"] == multiplier
        ]

        best = subset.loc[
            subset["validation_rmse_mean"].idxmin()
        ]

        best_rows.append(best)

    summary = pd.DataFrame(best_rows)

    summary = summary.sort_values(
        "validation_rmse_mean"
    )

    summary.to_csv(
        TABLES_DIR / "svt_common_stopping_summary.csv",
        index=False
    )

    return summary


def plot_validation_curves(curves):

    fig, ax = plt.subplots(figsize=(10, 6))

    for multiplier in TAU_MULTIPLIERS:

        subset = curves[
            curves["tau_multiplier"] == multiplier
        ]

        ax.plot(
            subset["iteration"],
            subset["validation_rmse_mean"],
            label=f"Tau = {multiplier}n"
        )

    ax.set_title(
        "Five-Fold SVT Validation RMSE"
    )

    ax.set_xlabel(
        "Post-initialization SVT iterations"
    )

    ax.set_ylabel(
        "Mean Validation RMSE"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    fig.savefig(
        FIGURES_DIR / "svt_cv_validation_curves.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_rank_curves(curves):

    fig, ax = plt.subplots(figsize=(10, 6))

    for multiplier in TAU_MULTIPLIERS:

        subset = curves[
            curves["tau_multiplier"] == multiplier
        ]

        ax.plot(
            subset["iteration"],
            subset["rank_mean"],
            label=f"Tau = {multiplier}n"
        )

    ax.set_title(
        "Average SVT Matrix Rank Across Five Folds"
    )

    ax.set_xlabel(
        "Post-initialization SVT iterations"
    )

    ax.set_ylabel(
        "Mean Matrix Rank"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    fig.savefig(
        FIGURES_DIR / "svt_cv_rank_curves.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():

    print("Loading five-fold SVT histories...")

    data = load_histories()

    print("Aggregating validation curves...")

    curves = aggregate_curves(data)

    print("Finding the best common stopping iteration...")

    summary = find_best_common_config(curves)

    plot_validation_curves(curves)
    plot_rank_curves(curves)

    print("\n" + "=" * 75)
    print("COMMON FIVE-FOLD SVT STOPPING RESULTS")
    print("=" * 75)

    print(
        summary[
            [
                "tau_multiplier",
                "iteration",
                "validation_rmse_mean",
                "validation_rmse_std",
                "validation_mae_mean",
                "rank_mean"
            ]
        ].to_string(
            index=False,
            float_format="%.4f"
        )
    )

    best = summary.iloc[0]

    print("\nBEST COMMON CONFIGURATION")

    print(
        f"Tau: {int(best['tau_multiplier'])}n"
    )

    print(
        f"Post-initialization iteration: "
        f"{int(best['iteration'])}"
    )

    print(
        f"Mean validation RMSE: "
        f"{best['validation_rmse_mean']:.4f}"
    )

    print("\nFigures saved:")
    print(FIGURES_DIR / "svt_cv_validation_curves.png")
    print(FIGURES_DIR / "svt_cv_rank_curves.png")

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
