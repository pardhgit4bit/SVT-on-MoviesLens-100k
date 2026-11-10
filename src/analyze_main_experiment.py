"""
Analyze the final MovieLens 100K holdout experiment.

Run locally:
    python src/analyze_main_experiment.py

Run with Weights & Biases:
    python src/analyze_main_experiment.py --wandb
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

SUMMARY_PATH = (
    TABLES_DIR / "main_experiment_final_test_results.csv"
)

PREDICTIONS_PATH = (
    TABLES_DIR / "main_experiment_final_test_predictions.csv"
)

BOOTSTRAP_REPLICATES = 2000
RANDOM_SEED = 42

PROJECT_NAME = "movielens-100k-matrix-completion"


# ============================================================
# 2. LOAD RESULTS
# ============================================================

def load_results():

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing results file: {SUMMARY_PATH}"
        )

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {PREDICTIONS_PATH}"
        )

    summary = pd.read_csv(SUMMARY_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    required_columns = {
        "actual_rating",
        "global_mean_prediction",
        "user_item_bias_prediction",
        "original_svt_prediction",
        "bias_adjusted_svt_prediction"
    }

    missing = required_columns.difference(
        predictions.columns
    )

    if missing:
        raise ValueError(
            f"Missing prediction columns: {missing}"
        )

    if predictions.empty:
        raise ValueError("Prediction file is empty.")

    return summary, predictions


# ============================================================
# 3. PAIRED BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

def calculate_bootstrap(
    predictions,
    n_bootstrap=BOOTSTRAP_REPLICATES
):

    actual = predictions[
        "actual_rating"
    ].to_numpy(dtype=np.float64)

    baseline = predictions[
        "user_item_bias_prediction"
    ].to_numpy(dtype=np.float64)

    models = {
        "Global Mean": "global_mean_prediction",
        "Original SVT": "original_svt_prediction",
        "Bias-Adjusted SVT": "bias_adjusted_svt_prediction"
    }

    baseline_squared_error = (baseline - actual) ** 2
    baseline_absolute_error = np.abs(baseline - actual)

    rng = np.random.default_rng(RANDOM_SEED)

    n = len(actual)
    results = []

    for model_name, column in models.items():

        candidate = predictions[
            column
        ].to_numpy(dtype=np.float64)

        candidate_squared_error = (candidate - actual) ** 2
        candidate_absolute_error = np.abs(candidate - actual)

        rmse_differences = np.empty(n_bootstrap)
        mae_differences = np.empty(n_bootstrap)

        for b in range(n_bootstrap):

            indices = rng.integers(
                0, n, size=n
            )

            candidate_rmse = np.sqrt(
                candidate_squared_error[indices].mean()
            )

            baseline_rmse = np.sqrt(
                baseline_squared_error[indices].mean()
            )

            candidate_mae = (
                candidate_absolute_error[indices].mean()
            )

            baseline_mae = (
                baseline_absolute_error[indices].mean()
            )

            rmse_differences[b] = (
                candidate_rmse - baseline_rmse
            )

            mae_differences[b] = (
                candidate_mae - baseline_mae
            )

        rmse_low, rmse_high = np.percentile(
            rmse_differences, [2.5, 97.5]
        )

        mae_low, mae_high = np.percentile(
            mae_differences, [2.5, 97.5]
        )

        results.append({

            "model_vs_user_item_bias": model_name,

            "rmse_difference": (
                np.sqrt(candidate_squared_error.mean())
                - np.sqrt(baseline_squared_error.mean())
            ),

            "rmse_ci_2_5": rmse_low,
            "rmse_ci_97_5": rmse_high,

            "mae_difference": (
                candidate_absolute_error.mean()
                - baseline_absolute_error.mean()
            ),

            "mae_ci_2_5": mae_low,
            "mae_ci_97_5": mae_high,

            "bootstrap_replicates": n_bootstrap,
            "random_seed": RANDOM_SEED
        })

    return pd.DataFrame(results)


# ============================================================
# 4. MODEL METRIC COMPARISON FIGURE
# ============================================================

def plot_model_metrics(summary):

    model_order = [
        "Global Mean",
        "User-Item Bias",
        "Original SVT",
        "Bias-Adjusted SVT"
    ]

    ordered = (
        summary.set_index("model")
        .loc[model_order]
        .reset_index()
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6)
    )

    metrics = [
        ("test_rmse", "Final Test RMSE"),
        ("test_mae", "Final Test MAE")
    ]

    for ax, (column, title) in zip(axes, metrics):

        bars = ax.bar(
            ordered["model"],
            ordered[column]
        )

        ax.set_title(title, fontsize=13)
        ax.set_ylabel("Error")
        ax.tick_params(
            axis="x",
            rotation=20
        )

        ax.grid(
            axis="y",
            alpha=0.25
        )

        ax.set_axisbelow(True)

        for bar, value in zip(
            bars, ordered[column]
        ):

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    fig.suptitle(
        "MovieLens 100K: Final Holdout Model Comparison",
        fontsize=15
    )

    fig.tight_layout()

    path = (
        FIGURES_DIR /
        "main_experiment_test_metrics.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# 5. RUNTIME COMPARISON FIGURE
# ============================================================

def plot_runtime(summary):

    model_order = [
        "Global Mean",
        "User-Item Bias",
        "Original SVT",
        "Bias-Adjusted SVT"
    ]

    ordered = (
        summary.set_index("model")
        .loc[model_order]
        .reset_index()
    )

    times = np.maximum(
        ordered["fit_seconds"].to_numpy(
            dtype=np.float64
        ),
        1e-5
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.bar(
        ordered["model"],
        times
    )

    ax.set_yscale("log")

    ax.set_title(
        "Final Holdout Runtime Comparison",
        fontsize=14
    )

    ax.set_ylabel(
        "Fit and prediction time (seconds, log scale)"
    )

    ax.tick_params(
        axis="x",
        rotation=20
    )

    ax.grid(
        axis="y",
        which="both",
        alpha=0.25
    )

    ax.set_axisbelow(True)

    for bar, value in zip(
        bars, ordered["fit_seconds"]
    ):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4g}s",
            ha="center",
            va="bottom"
        )

    fig.tight_layout()

    path = (
        FIGURES_DIR /
        "main_experiment_runtime.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# 6. BOOTSTRAP CONFIDENCE INTERVAL FIGURE
# ============================================================

def plot_bootstrap(bootstrap):

    names = bootstrap[
        "model_vs_user_item_bias"
    ].tolist()

    positions = np.arange(len(names))

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6)
    )

    metrics = [

        (
            axes[0],
            "rmse_difference",
            "rmse_ci_2_5",
            "rmse_ci_97_5",
            "RMSE Difference vs User-Item Bias"
        ),

        (
            axes[1],
            "mae_difference",
            "mae_ci_2_5",
            "mae_ci_97_5",
            "MAE Difference vs User-Item Bias"
        )
    ]

    for (
        ax,
        center_col,
        low_col,
        high_col,
        title
    ) in metrics:

        centers = bootstrap[
            center_col
        ].to_numpy()

        lows = bootstrap[
            low_col
        ].to_numpy()

        highs = bootstrap[
            high_col
        ].to_numpy()

        errors = np.vstack([
            centers - lows,
            highs - centers
        ])

        ax.errorbar(
            centers,
            positions,
            xerr=errors,
            fmt="o",
            capsize=5
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1
        )

        ax.set_yticks(
            positions,
            names
        )

        ax.invert_yaxis()

        ax.set_title(title)

        ax.set_xlabel(
            "Difference (negative favors compared model)"
        )

        ax.grid(
            axis="x",
            alpha=0.25
        )

        ax.set_axisbelow(True)

    fig.suptitle(
        "Paired Bootstrap 95% Confidence Intervals"
    )

    fig.tight_layout()

    path = (
        FIGURES_DIR /
        "main_experiment_paired_bootstrap.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# 7. WEIGHTS & BIASES LOGGING
# ============================================================

def log_to_wandb(
    summary,
    bootstrap,
    figure_paths,
    bootstrap_path
):

    try:
        import wandb

    except ImportError as exc:

        raise RuntimeError(
            "W&B is not installed. Run: "
            "python -m pip install wandb"
        ) from exc

    config = {

        "dataset": "MovieLens 100K",

        "protocol": (
            "80% development / 20% untouched final test"
        ),

        "bias_adjusted_tau_multiplier": 10,
        "bias_adjusted_delta": 1.0,
        "bias_adjusted_iterations": 7,

        "original_svt_tau": 9430.0,
        "original_svt_delta": 1.0,
        "original_svt_iterations": 425,

        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": RANDOM_SEED,

        "test_used_for_model_selection": False
    }

    with wandb.init(
        project=PROJECT_NAME,
        name="final-holdout-comparison",
        config=config
    ) as run:

        metrics = {}

        for _, row in summary.iterrows():

            key = (
                str(row["model"])
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            metrics[f"{key}/test_rmse"] = float(
                row["test_rmse"]
            )

            metrics[f"{key}/test_mae"] = float(
                row["test_mae"]
            )

            metrics[f"{key}/fit_seconds"] = float(
                row["fit_seconds"]
            )

            if pd.notna(row.get("rank")):

                metrics[f"{key}/rank"] = int(
                    row["rank"]
                )

        run.log(metrics)

        run.log({

            "figures/model_metrics": wandb.Image(
                str(figure_paths[0])
            ),

            "figures/runtime": wandb.Image(
                str(figure_paths[1])
            ),

            "figures/paired_bootstrap": wandb.Image(
                str(figure_paths[2])
            )
        })

        # Upload result tables as a W&B artifact.
        artifact = wandb.Artifact(
            "movielens-final-holdout-results",
            type="evaluation"
        )

        artifact.add_file(str(SUMMARY_PATH))
        artifact.add_file(str(PREDICTIONS_PATH))
        artifact.add_file(str(bootstrap_path))

        run.log_artifact(artifact)

    print("\nW&B logging completed successfully.")


# ============================================================
# 8. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Upload results and figures to Weights & Biases"
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=BOOTSTRAP_REPLICATES
    )

    args = parser.parse_args()

    if args.bootstrap < 100:
        raise ValueError(
            "Use at least 100 bootstrap replicates."
        )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary, predictions = load_results()

    print("\nLoaded final experiment results.")

    # --------------------------------------------------------
    # Bootstrap analysis
    # --------------------------------------------------------

    bootstrap = calculate_bootstrap(
        predictions,
        n_bootstrap=args.bootstrap
    )

    bootstrap_path = (
        TABLES_DIR /
        "main_experiment_paired_bootstrap_vs_bias.csv"
    )

    bootstrap.to_csv(
        bootstrap_path,
        index=False
    )

    # --------------------------------------------------------
    # Generate figures
    # --------------------------------------------------------

    figure_1 = plot_model_metrics(summary)
    figure_2 = plot_runtime(summary)
    figure_3 = plot_bootstrap(bootstrap)

    figure_paths = [
        figure_1,
        figure_2,
        figure_3
    ]

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("FINAL TEST METRICS")
    print("=" * 90)

    print(
        summary[
            [
                "model",
                "test_rmse",
                "test_mae",
                "fit_seconds",
                "rank"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\n" + "=" * 90)
    print("PAIRED BOOTSTRAP: MODEL MINUS USER-ITEM BIAS")
    print("Negative differences favor the compared model.")
    print("=" * 90)

    print(
        bootstrap.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # --------------------------------------------------------
    # Calculate proposed-model improvements
    # --------------------------------------------------------

    baseline = summary.loc[
        summary["model"] == "User-Item Bias"
    ].iloc[0]

    proposed = summary.loc[
        summary["model"] == "Bias-Adjusted SVT"
    ].iloc[0]

    rmse_improvement = (
        (baseline["test_rmse"] - proposed["test_rmse"])
        / baseline["test_rmse"]
        * 100
    )

    mae_improvement = (
        (baseline["test_mae"] - proposed["test_mae"])
        / baseline["test_mae"]
        * 100
    )

    print("\nPROPOSED MODEL IMPROVEMENT")

    print(
        f"RMSE reduction: {rmse_improvement:.3f}%"
    )

    print(
        f"MAE reduction : {mae_improvement:.3f}%"
    )

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    print("\nSaved bootstrap results:")
    print(bootstrap_path)

    print("\nSaved figures:")

    for path in figure_paths:
        print(path)

    # --------------------------------------------------------
    # Optional W&B
    # --------------------------------------------------------

    if args.wandb:

        log_to_wandb(
            summary,
            bootstrap,
            figure_paths,
            bootstrap_path
        )


if __name__ == "__main__":
    main()