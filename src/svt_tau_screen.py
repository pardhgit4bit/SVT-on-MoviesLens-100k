
# src/svt_tau_screen.py

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)


# Preliminary threshold screening
N = min(N_USERS, N_ITEMS)

TAU_MULTIPLIERS = [6, 7, 8, 9]

DELTA = 1.0
MAX_ITERATIONS = 500


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


def main():

    train = pd.read_csv(
        SPLIT_DIR / "fold_1_train.csv"
    )

    validation = pd.read_csv(
        SPLIT_DIR / "fold_1_validation.csv"
    )

    records = []

    for multiplier in TAU_MULTIPLIERS:

        tau = multiplier * N

        logging.info("=" * 55)
        logging.info(
            "Testing tau = %d * %d = %.2f",
            multiplier, N, tau
        )

        model = SingularValueThresholding(
            tau=tau,
            delta=DELTA,
            epsilon=EPSILON,
            max_iterations=MAX_ITERATIONS,
            increment=SVD_INCREMENT,
            n_users=N_USERS,
            n_items=N_ITEMS
        )

        model.fit(
            train,
            validation=validation
        )

        record = {
            "tau_multiplier": multiplier,
            "tau": tau,
            "delta": DELTA,
            "best_iteration": model.best_validation_iteration_,
            "best_rank": model.best_validation_rank_,
            "best_validation_rmse": model.best_validation_rmse_,
            "best_validation_mae": model.best_validation_mae_,
            "final_rank": model.final_rank_,
            "final_observed_error": model.final_relative_error_,
            "iterations": model.n_iterations_,
            "converged": model.converged_,
            "time_seconds": model.execution_time_
        }

        records.append(record)

        # Save after each candidate, so completed results
        # are preserved if a later experiment fails.
        results = pd.DataFrame(records)

        results.to_csv(
            TABLES_DIR / "svt_tau_screening.csv",
            index=False
        )

        logging.info(
            "BEST | tau=%d | iteration=%d | "
            "RMSE=%.4f | rank=%d",
            tau,
            model.best_validation_iteration_,
            model.best_validation_rmse_,
            model.best_validation_rank_
        )

    results = pd.DataFrame(records)

    results = results.sort_values(
        "best_validation_rmse"
    )

    print("\n" + "=" * 75)
    print("SVT THRESHOLD SCREENING RESULTS")
    print("=" * 75)

    print(
        results.to_string(
            index=False,
            float_format="%.4f"
        )
    )

    print("\nBest threshold:")
    print(results.iloc[0][[
        "tau",
        "best_iteration",
        "best_validation_rmse",
        "best_validation_mae",
        "best_rank"
    ]].to_string())

    print("\nSaved to:")
    print(TABLES_DIR / "svt_tau_screening.csv")


if __name__ == "__main__":
    main()
