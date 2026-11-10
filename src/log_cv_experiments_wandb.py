
"""
Log saved MovieLens development experiments to Weights & Biases.

Reads saved CSV/PNG outputs only. It does NOT retrain models or access the
final holdout test set.

Run from the project root:
    python src/log_cv_experiments_wandb.py
"""

from pathlib import Path
import pandas as pd
import wandb


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

PROJECT = "movielens-100k-matrix-completion"
TAUS = [5, 10, 15, 20]
FOLDS = [1, 2, 3, 4, 5]
SELECTED_TAU = 10


def required(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Run the relevant experiment first."
        )
    return path


def load_outputs():
    fold_path = required(
        TABLES / "bias_adjusted_svt_cv_fold_results.csv"
    )
    folds = pd.read_csv(fold_path)

    needed = {
        "fold", "tau_multiplier", "best_iteration", "best_rank",
        "best_validation_rmse", "best_validation_mae",
        "execution_time_seconds",
    }
    missing = needed.difference(folds.columns)
    if missing:
        raise ValueError(f"Fold-results CSV missing columns: {sorted(missing)}")

    folds["fold"] = folds["fold"].astype(int)
    folds["tau_multiplier"] = folds["tau_multiplier"].astype(int)

    expected = {(tau, fold) for tau in TAUS for fold in FOLDS}
    found = set(zip(folds.tau_multiplier, folds.fold))
    if expected - found:
        raise ValueError(
            f"Threshold results are incomplete: {sorted(expected - found)}"
        )

    histories = {}
    for tau in TAUS:
        for fold in FOLDS:
            path = required(
                TABLES / f"bias_adjusted_svt_tau_{tau}_fold_{fold}_history.csv"
            )
            history = pd.read_csv(path)
            cols = {
                "iteration", "rank", "relative_observed_error",
                "validation_rmse", "validation_mae",
            }
            missing = cols.difference(history.columns)
            if missing:
                raise ValueError(
                    f"{path.name} missing columns: {sorted(missing)}"
                )
            histories[(tau, fold)] = history

    curve_path = required(
        TABLES / f"bias_adjusted_svt_cv_mean_curve_tau_{SELECTED_TAU}.csv"
    )
    curve = pd.read_csv(curve_path)

    return folds, histories, curve, fold_path, curve_path


def threshold_summary(folds):
    return (
        folds.groupby("tau_multiplier")
        .agg(
            mean_rmse=("best_validation_rmse", "mean"),
            std_rmse=("best_validation_rmse", "std"),
            mean_mae=("best_validation_mae", "mean"),
            std_mae=("best_validation_mae", "std"),
            mean_best_iteration=("best_iteration", "mean"),
            mean_best_rank=("best_rank", "mean"),
            mean_final_rank=("final_rank", "mean"),
            mean_runtime_seconds=("execution_time_seconds", "mean"),
        )
        .reset_index()
        .sort_values("tau_multiplier")
    )


def log_curves(run, histories, tau_values):
    """Log per-fold validation curves, rank and observed residual by iteration."""
    selected = {
        key: value.set_index("iteration")
        for key, value in histories.items()
        if key[0] in tau_values
    }
    max_step = max(int(frame.index.max()) for frame in selected.values())

    for iteration in range(1, max_step + 1):
        payload = {}
        for (tau, fold), frame in selected.items():
            if iteration not in frame.index:
                continue
            row = frame.loc[iteration]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            prefix = f"tau_{tau}/fold_{fold}"
            for column in (
                "validation_rmse", "validation_mae",
                "rank", "relative_observed_error",
            ):
                value = row[column]
                if pd.notna(value):
                    payload[f"{prefix}/{column}"] = float(value)
        if payload:
            run.log(payload, step=iteration)


def log_artifact(run, name, paths, description):
    artifact = wandb.Artifact(
        name=name,
        type="development-experiment-results",
        description=description,
    )
    for path in paths:
        if path.exists():
            artifact.add_file(str(path))
    run.log_artifact(artifact)


def log_selected_cv(folds, histories, curve, fold_path, curve_path):
    chosen = (
        folds[folds.tau_multiplier == SELECTED_TAU]
        .sort_values("fold")
        .copy()
    )
    if len(chosen) != 5:
        raise ValueError("Expected five fold rows for selected tau=10.")

    common = curve.loc[curve.mean_validation_rmse.idxmin()]

    with wandb.init(
        project=PROJECT,
        name="bias-adjusted-svt-five-fold-cv-tau10",
        group="bias-adjusted-svt-development",
        job_type="five-fold-cross-validation",
        tags=["bias-adjusted-svt", "five-fold-cv", "selected-model"],
        config={
            "dataset": "MovieLens 100K",
            "tau_multiplier": SELECTED_TAU,
            "delta": 1.0,
            "selected_common_iteration": int(common.iteration),
            "folds": 5,
            "bias_regularization": 10.0,
            "bias_iterations": 20,
            "selection_rule": "minimum mean validation RMSE across five folds",
        },
    ) as run:

        # First log all iteration curves (steps 1..30), then summary/table data.
        log_curves(run, histories, [SELECTED_TAU])

        run.log({
            "cv/best_per_fold_mean_rmse": float(
                chosen.best_validation_rmse.mean()
            ),
            "cv/best_per_fold_std_rmse": float(
                chosen.best_validation_rmse.std()
            ),
            "cv/best_per_fold_mean_mae": float(
                chosen.best_validation_mae.mean()
            ),
            "cv/common_iteration": int(common.iteration),
            "cv/common_iteration_mean_rmse": float(
                common.mean_validation_rmse
            ),
            "cv/common_iteration_std_rmse": float(
                common.std_validation_rmse
            ),
            "cv/common_iteration_mean_mae": float(
                common.mean_validation_mae
            ),
            "cv/common_iteration_mean_rank": float(common.mean_rank),
        })

        run.log({
            "tables/fold_results": wandb.Table(dataframe=chosen),
            "tables/common_iteration_curve": wandb.Table(dataframe=curve),
        })

        history_files = [
            TABLES / f"bias_adjusted_svt_tau_{SELECTED_TAU}_fold_{f}_history.csv"
            for f in FOLDS
        ]
        log_artifact(
            run,
            "bias-adjusted-svt-five-fold-cv-tau10",
            [fold_path, curve_path, *history_files],
            "Selected tau=10 five-fold CV and iteration histories.",
        )

    print("W&B run logged: selected tau=10 five-fold CV.")


def log_threshold_screen(folds, histories, fold_path):
    summary = threshold_summary(folds)

    with wandb.init(
        project=PROJECT,
        name="bias-adjusted-svt-threshold-screening",
        group="bias-adjusted-svt-development",
        job_type="threshold-screening",
        tags=["bias-adjusted-svt", "threshold-screening", "five-fold-cv"],
        config={
            "dataset": "MovieLens 100K",
            "tau_multipliers": TAUS,
            "delta": 1.0,
            "max_iterations": 30,
            "folds": 5,
            "selection_metric": "mean validation RMSE",
        },
    ) as run:

        # Log all 20 per-fold learning curves with iteration as the step.
        log_curves(run, histories, TAUS)

        metrics = {}
        for _, row in summary.iterrows():
            tau = int(row.tau_multiplier)
            for column in (
                "mean_rmse", "std_rmse", "mean_mae", "std_mae",
                "mean_best_iteration", "mean_best_rank",
                "mean_final_rank", "mean_runtime_seconds",
            ):
                metrics[f"threshold/tau_{tau}/{column}"] = float(row[column])
        run.log(metrics)

        run.log({
            "tables/all_fold_results": wandb.Table(dataframe=folds),
            "tables/threshold_summary": wandb.Table(dataframe=summary),
        })

        histories_paths = [
            TABLES / f"bias_adjusted_svt_tau_{tau}_fold_{fold}_history.csv"
            for tau in TAUS for fold in FOLDS
        ]
        log_artifact(
            run,
            "bias-adjusted-svt-threshold-screening",
            [fold_path, *histories_paths],
            "Four tau multipliers across five development folds, including histories.",
        )

    print("W&B run logged: four-threshold screening.")


def log_original_svt_cv_if_present():
    # These are the outputs created by the earlier Original SVT CV analysis.
    candidate_csvs = [
        TABLES / "svt_cv_iteration_summary.csv",
        TABLES / "svt_cv_fold_results.csv",
        TABLES / "svt_cv_summary.csv",
    ]
    csvs = [path for path in candidate_csvs if path.exists()]
    histories = sorted(TABLES.glob("svt*fold*history*.csv"))
    figures = [
        path for path in [
            FIGURES / "svt_cv_validation_curves.png",
            FIGURES / "svt_cv_rank_curves.png",
        ] if path.exists()
    ]

    if not csvs and not histories and not figures:
        print("No saved Original SVT CV outputs detected; skipping that optional run.")
        return

    with wandb.init(
        project=PROJECT,
        name="original-svt-five-fold-cv",
        group="original-svt-development",
        job_type="five-fold-cross-validation",
        tags=["original-svt", "five-fold-cv"],
        config={
            "dataset": "MovieLens 100K",
            "model": "Original SVT",
            "source": "Previously completed development-only cross-validation",
            "final_test_used": False,
        },
    ) as run:

        for path in csvs:
            try:
                run.log({
                    f"tables/{path.stem}": wandb.Table(
                        dataframe=pd.read_csv(path)
                    )
                })
            except Exception as exc:
                print(f"Skipping table {path.name}: {exc}")

        for path in figures:
            run.log({f"figures/{path.stem}": wandb.Image(str(path))})

        log_artifact(
            run,
            "original-svt-development-cv",
            [*csvs, *histories, *figures],
            "Saved Original SVT development-only five-fold CV outputs.",
        )

    print("W&B run logged: available Original SVT CV outputs.")


def main():
    print("=" * 78)
    print("LOGGING DEVELOPMENT EXPERIMENTS TO W&B")
    print("=" * 78)

    folds, histories, curve, fold_path, curve_path = load_outputs()

    print("\nThreshold summary:")
    print(
        threshold_summary(folds).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    log_selected_cv(
        folds, histories, curve, fold_path, curve_path
    )
    log_threshold_screen(
        folds, histories, fold_path
    )
    log_original_svt_cv_if_present()

    print("\nDevelopment experiment logging completed.")
    print(f"Project: {PROJECT}")
    print("No training was repeated; final test data was not accessed.")


if __name__ == "__main__":
    main()
