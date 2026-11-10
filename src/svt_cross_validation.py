
# src/svt_cross_validation.py

from pathlib import Path
import logging

import pandas as pd

from svt import (
    SingularValueThresholding,
    N_USERS,
    N_ITEMS,
    EPSILON,
    SVD_INCREMENT
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5

# Candidates selected from preliminary Fold 1 screening.
TAU_MULTIPLIERS = [8, 9, 10]

DELTA = 1.0
MAX_ITERATIONS = 500

# Threshold scale used in this experiment.
N = min(N_USERS, N_ITEMS)


# Suppress per-iteration INFO logs during the 15 experiments.
logging.getLogger().setLevel(logging.WARNING)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_cross_validation():

    fold_records = []

    total_experiments = len(TAU_MULTIPLIERS) * N_FOLDS
    experiment_number = 0

    for multiplier in TAU_MULTIPLIERS:

        tau = multiplier * N

        print("\n" + "=" * 75)
        print(f"TESTING TAU = {multiplier}n = {tau:.2f}")
        print("=" * 75)

        for fold in range(1, N_FOLDS + 1):

            experiment_number += 1

            print(
                f"\nExperiment {experiment_number}/{total_experiments}"
                f" | Tau: {tau:.0f} | Fold: {fold}"
            )

            # ------------------------------------------------
            # Load fold
            # ------------------------------------------------

            train = pd.read_csv(
                SPLIT_DIR / f"fold_{fold}_train.csv"
            )

            validation = pd.read_csv(
                SPLIT_DIR / f"fold_{fold}_validation.csv"
            )

            # ------------------------------------------------
            # Initialize model
            # ------------------------------------------------

            model = SingularValueThresholding(

                tau=tau,

                delta=DELTA,

                epsilon=EPSILON,

                max_iterations=MAX_ITERATIONS,

                increment=SVD_INCREMENT,

                n_users=N_USERS,

                n_items=N_ITEMS

            )

            # ------------------------------------------------
            # Train and monitor validation performance
            # ------------------------------------------------

            model.fit(
                train,
                validation=validation
            )

            # ------------------------------------------------
            # Extract iteration history
            # ------------------------------------------------

            history = pd.DataFrame(
                model.history_
            )

            history_path = (
                TABLES_DIR /
                f"svt_tau_{multiplier}_fold_{fold}_history.csv"
            )

            history.to_csv(
                history_path,
                index=False
            )

            # Best validation metrics
            best_rmse = model.best_validation_rmse_
            best_mae = model.best_validation_mae_

            # Validation metrics at the final iteration
            final_validation_rmse = (
                history.iloc[-1]["validation_rmse"]
            )

            final_validation_mae = (
                history.iloc[-1]["validation_mae"]
            )

            # ------------------------------------------------
            # Store experiment results
            # ------------------------------------------------

            record = {

                "tau_multiplier": multiplier,

                "tau": tau,

                "delta": DELTA,

                "fold": fold,

                "best_iteration":
                    model.best_validation_iteration_,

                "best_rank":
                    model.best_validation_rank_,

                "best_validation_rmse":
                    best_rmse,

                "best_validation_mae":
                    best_mae,

                "final_validation_rmse":
                    final_validation_rmse,

                "final_validation_mae":
                    final_validation_mae,

                "final_rank":
                    model.final_rank_,

                "relative_observed_error":
                    model.final_relative_error_,

                "iterations":
                    model.n_iterations_,

                "converged":
                    model.converged_,

                "execution_time_seconds":
                    model.execution_time_

            }

            fold_records.append(record)

            # Save after every completed fold.
            fold_results = pd.DataFrame(
                fold_records
            )

            fold_results.to_csv(
                TABLES_DIR / "svt_cv_fold_results.csv",
                index=False
            )

            print(
                f"Best RMSE: {best_rmse:.4f}"
                f" | Best MAE: {best_mae:.4f}"
                f" | Best iteration: "
                f"{model.best_validation_iteration_}"
                f" | Rank: {model.best_validation_rank_}"
                f" | Time: {model.execution_time_:.2f}s"
            )

    # ========================================================
    # AGGREGATE FIVE-FOLD RESULTS
    # ========================================================

    results = pd.DataFrame(
        fold_records
    )

    summary = (

        results.groupby(
            ["tau_multiplier", "tau"]
        )

        .agg(

            RMSE_mean=(
                "best_validation_rmse", "mean"
            ),

            RMSE_std=(
                "best_validation_rmse", "std"
            ),

            MAE_mean=(
                "best_validation_mae", "mean"
            ),

            MAE_std=(
                "best_validation_mae", "std"
            ),

            Mean_best_iteration=(
                "best_iteration", "mean"
            ),

            Mean_best_rank=(
                "best_rank", "mean"
            ),

            Mean_final_rank=(
                "final_rank", "mean"
            ),

            Mean_execution_time=(
                "execution_time_seconds", "mean"
            ),

            Converged_folds=(
                "converged", "sum"
            )

        )

        .reset_index()

        .sort_values("RMSE_mean")

    )

    summary.to_csv(
        TABLES_DIR / "svt_cv_summary.csv",
        index=False
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 85)
    print("FIVE-FOLD SVT CROSS-VALIDATION RESULTS")
    print("=" * 85)

    print(
        summary.to_string(
            index=False,
            float_format="%.4f"
        )
    )

    # --------------------------------------------------------
    # Best candidate
    # --------------------------------------------------------

    best = summary.iloc[0]

    print("\n" + "=" * 65)
    print("BEST SVT CONFIGURATION")
    print("=" * 65)

    print(f"Tau multiplier       : {int(best['tau_multiplier'])}n")
    print(f"Tau                  : {best['tau']:.2f}")
    print(f"Delta                : {DELTA:.2f}")

    print(
        f"Mean CV RMSE         : "
        f"{best['RMSE_mean']:.4f} "
        f"+/- {best['RMSE_std']:.4f}"
    )

    print(
        f"Mean CV MAE          : "
        f"{best['MAE_mean']:.4f} "
        f"+/- {best['MAE_std']:.4f}"
    )

    print(
        f"Mean best iteration  : "
        f"{best['Mean_best_iteration']:.1f}"
    )

    print(
        f"Mean best rank       : "
        f"{best['Mean_best_rank']:.1f}"
    )

    print("\nSaved files:")

    print(TABLES_DIR / "svt_cv_fold_results.csv")
    print(TABLES_DIR / "svt_cv_summary.csv")

    print("\nCross-validation completed.")


if __name__ == "__main__":

    run_cross_validation()
