"""
Aggregate the five validation curves for Bias-Adjusted SVT.

Run from the project root:
    python src/analyze_bias_adjusted_svt_cv.py

This selects one common iteration by minimizing the mean validation RMSE
across the five folds, rather than averaging five fold-specific minima.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

TAU_MULTIPLIER = 10
EXPECTED_FOLDS = 5


def main():
    histories = []

    for fold in range(1, EXPECTED_FOLDS + 1):
        path = (
            TABLES_DIR
            / f"bias_adjusted_svt_tau_{TAU_MULTIPLIER}_fold_{fold}_history.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing history file: {path}\n"
                "Run the five-fold threshold experiment first."
            )

        history = pd.read_csv(path)

        required = {
            "iteration",
            "validation_rmse",
            "validation_mae",
            "rank",
        }
        missing = required.difference(history.columns)

        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {sorted(missing)}"
            )

        history["fold"] = fold
        histories.append(history)

    all_histories = pd.concat(histories, ignore_index=True)

    # A common iteration must be represented in all five folds.
    curve = (
        all_histories
        .groupby("iteration")
        .agg(
            number_of_folds=("fold", "nunique"),
            mean_validation_rmse=("validation_rmse", "mean"),
            std_validation_rmse=("validation_rmse", "std"),
            mean_validation_mae=("validation_mae", "mean"),
            std_validation_mae=("validation_mae", "std"),
            mean_rank=("rank", "mean"),
        )
        .reset_index()
    )

    curve = curve[
        curve["number_of_folds"] == EXPECTED_FOLDS
    ].copy()

    if curve.empty:
        raise RuntimeError(
            "No iteration has validation results from all five folds."
        )

    selected = curve.loc[
        curve["mean_validation_rmse"].idxmin()
    ]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = (
        TABLES_DIR
        / f"bias_adjusted_svt_cv_mean_curve_tau_{TAU_MULTIPLIER}.csv"
    )
    curve.to_csv(csv_path, index=False)

    # Plot the common five-fold validation curves.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(
        curve["iteration"],
        curve["mean_validation_rmse"],
        marker="o",
        markersize=3,
        label="Mean validation RMSE",
    )
    axes[0].fill_between(
        curve["iteration"],
        curve["mean_validation_rmse"] - curve["std_validation_rmse"],
        curve["mean_validation_rmse"] + curve["std_validation_rmse"],
        alpha=0.18,
        label="±1 standard deviation",
    )
    axes[0].axvline(
        int(selected["iteration"]),
        linestyle="--",
        label=f"Selected iteration: {int(selected['iteration'])}",
    )
    axes[0].set_title("Bias-Adjusted SVT: Validation RMSE")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        curve["iteration"],
        curve["mean_validation_mae"],
        marker="o",
        markersize=3,
        label="Mean validation MAE",
    )
    axes[1].fill_between(
        curve["iteration"],
        curve["mean_validation_mae"] - curve["std_validation_mae"],
        curve["mean_validation_mae"] + curve["std_validation_mae"],
        alpha=0.18,
        label="±1 standard deviation",
    )
    axes[1].axvline(
        int(selected["iteration"]),
        linestyle="--",
        label=f"Selected iteration: {int(selected['iteration'])}",
    )
    axes[1].set_title("Bias-Adjusted SVT: Validation MAE")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("MAE")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle(
        f"Five-Fold Mean Validation Curves (Tau Multiplier = {TAU_MULTIPLIER})"
    )
    fig.tight_layout()

    figure_path = (
        FIGURES_DIR
        / f"bias_adjusted_svt_cv_mean_curve_tau_{TAU_MULTIPLIER}.png"
    )
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 76)
    print("COMMON ITERATION SELECTION")
    print("=" * 76)
    print(f"Tau multiplier              : {TAU_MULTIPLIER}")
    print(f"Selected common iteration   : {int(selected['iteration'])}")
    print(
        "Mean validation RMSE        : "
        f"{selected['mean_validation_rmse']:.6f}"
    )
    print(
        "Std. validation RMSE        : "
        f"{selected['std_validation_rmse']:.6f}"
    )
    print(
        "Mean validation MAE         : "
        f"{selected['mean_validation_mae']:.6f}"
    )
    print(
        "Mean validation rank        : "
        f"{selected['mean_rank']:.2f}"
    )
    print("\nSaved:")
    print(csv_path)
    print(figure_path)


if __name__ == "__main__":
    main()
