"""
Final holdout experiment for MovieLens 100K.

Protocol:
- Fit on data/splits/development.csv (80%).
- Evaluate once on data/splits/final_test.csv (20%).
- Do not use final_test.csv for model selection, early stopping, or tuning.

Locked settings selected from development-only cross-validation:
- Original SVT: tau = 10 * 943, delta = 1, 425 iterations.
- Bias-Adjusted SVT: tau multiplier = 10, delta = 1, 7 iterations.

Run from the project root:
    python src/main_experiment.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error

from baselines import UserItemBiasBaseline
from bias_adjusted_svt import BiasAdjustedSVT
from svt import (
    compute_leading_singular_triplets,
    predict_low_rank_entries,
    N_USERS,
    N_ITEMS,
    SVD_INCREMENT,
    EPSILON,
)


# ============================================================
# 1. LOCKED EXPERIMENT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

RATING_MIN = 1.0
RATING_MAX = 5.0

# Original SVT: selected from its own five-fold development CV.
ORIGINAL_SVT_TAU_MULTIPLIER = 10.0
ORIGINAL_SVT_TAU = ORIGINAL_SVT_TAU_MULTIPLIER * N_USERS
ORIGINAL_SVT_DELTA = 1.0
ORIGINAL_SVT_ITERATIONS = 425

# Proposed bias-adjusted extension: selected from its five-fold CV.
BIAS_SVT_TAU_MULTIPLIER = 10.0
BIAS_SVT_DELTA = 1.0
BIAS_SVT_ITERATIONS = 7

BIAS_REGULARIZATION = 10.0
BIAS_ITERATIONS = 20

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)


# ============================================================
# 2. DATA AND METRIC HELPERS
# ============================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    development_path = SPLIT_DIR / "development.csv"
    final_test_path = SPLIT_DIR / "final_test.csv"

    if not development_path.exists():
        raise FileNotFoundError(
            f"Missing {development_path}. Run src/preprocessing.py first."
        )
    if not final_test_path.exists():
        raise FileNotFoundError(
            f"Missing {final_test_path}. Run src/preprocessing.py first."
        )

    development = pd.read_csv(development_path)
    final_test = pd.read_csv(final_test_path)

    required = {"user_id", "item_id", "rating"}
    for name, frame in (
        ("development", development),
        ("final_test", final_test),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} data is missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{name} data is empty.")
        if frame.duplicated(["user_id", "item_id"]).any():
            raise ValueError(f"Duplicate user-item pairs found in {name} data.")
        if not np.isfinite(frame["rating"].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Non-finite ratings found in {name} data.")

    # MovieLens 100K IDs are 1-based and the models use fixed matrix dimensions.
    for name, frame in (
        ("development", development),
        ("final_test", final_test),
    ):
        users = frame["user_id"].to_numpy(dtype=np.int64)
        items = frame["item_id"].to_numpy(dtype=np.int64)
        if np.any((users < 1) | (users > N_USERS)):
            raise ValueError(f"User ID outside 1..{N_USERS} in {name} data.")
        if np.any((items < 1) | (items > N_ITEMS)):
            raise ValueError(f"Item ID outside 1..{N_ITEMS} in {name} data.")

    # Fail fast if the split files accidentally overlap.
    dev_pairs = set(
        zip(
            development["user_id"].to_numpy(dtype=np.int64),
            development["item_id"].to_numpy(dtype=np.int64),
        )
    )
    test_pairs = set(
        zip(
            final_test["user_id"].to_numpy(dtype=np.int64),
            final_test["item_id"].to_numpy(dtype=np.int64),
        )
    )
    overlap = dev_pairs.intersection(test_pairs)
    if overlap:
        raise ValueError(
            f"Development and final test contain {len(overlap)} overlapping "
            "user-item pairs."
        )

    return development, final_test


def arrays(frame: pd.DataFrame):
    users = frame["user_id"].to_numpy(dtype=np.int64)
    items = frame["item_id"].to_numpy(dtype=np.int64)
    actual = frame["rating"].to_numpy(dtype=np.float64)
    return users, items, actual


def clipped(predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.float64)
    if not np.isfinite(predictions).all():
        raise ValueError("A model produced non-finite predictions.")
    return np.clip(predictions, RATING_MIN, RATING_MAX)


def evaluate(
    model_name: str,
    predictions: np.ndarray,
    actual: np.ndarray,
    fit_seconds: float,
    rank: int | None = None,
    details: dict | None = None,
) -> dict:
    predictions = clipped(predictions)
    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
    mae = float(mean_absolute_error(actual, predictions))

    result = {
        "model": model_name,
        "test_rmse": rmse,
        "test_mae": mae,
        "fit_seconds": float(fit_seconds),
        "rank": rank,
    }
    if details:
        result.update(details)

    print(
        f"{model_name:24s} | RMSE: {rmse:.6f} | MAE: {mae:.6f} "
        f"| fit: {fit_seconds:.2f}s"
        + (f" | rank: {rank}" if rank is not None else "")
    )
    return result


# ============================================================
# 3. ORIGINAL, PAPER-BASED SVT AT FIXED ITERATION 425
# ============================================================

def fit_predict_original_svt(
    development: pd.DataFrame,
    test_users: np.ndarray,
    test_items: np.ndarray,
) -> tuple[np.ndarray, float, int, float]:
    """
    Run original SVT on raw observed ratings.

    The recurrence and adaptive partial-SVD strategy use the same helpers
    as src/svt.py. The final-test coordinates are used only after all
    425 iterations have completed.
    """
    start_time = time.perf_counter()

    train = development.sort_values(
        ["user_id", "item_id"]
    ).reset_index(drop=True)

    users, items, ratings = arrays(train)
    rows = users - 1
    cols = items - 1

    observed = csr_matrix(
        (ratings, (rows, cols)),
        shape=(N_USERS, N_ITEMS),
        dtype=np.float64,
    )

    if observed.nnz != len(train):
        raise RuntimeError(
            "Sparse matrix construction changed the number of observations."
        )

    # Spectral norm of P_Omega(M), used by the paper's Y0 initialization.
    _, leading_values, _, initial_backend = (
        compute_leading_singular_triplets(observed, k=1)
    )
    spectral_norm = float(leading_values[0])
    if not np.isfinite(spectral_norm) or spectral_norm <= 0:
        raise ValueError("Invalid spectral norm for original SVT.")

    tau = ORIGINAL_SVT_TAU
    delta = ORIGINAL_SVT_DELTA
    k0 = max(1, int(np.ceil(tau / (delta * spectral_norm))))

    # Y0 = k0 * delta * P_Omega(M), stored only on observed entries.
    Y = observed.copy()
    Y.data *= k0 * delta

    previous_rank = 0
    U = np.empty((N_USERS, 0), dtype=np.float64)
    shrunk_values = np.empty(0, dtype=np.float64)
    VT = np.empty((0, N_ITEMS), dtype=np.float64)
    current_rank = 0
    final_backend = initial_backend

    for iteration in range(1, ORIGINAL_SVT_ITERATIONS + 1):
        # Adaptive partial SVD: find all singular values strictly > tau.
        requested = previous_rank + 1

        while True:
            requested = min(requested, min(Y.shape))
            U_all, singular_values, VT_all, backend = (
                compute_leading_singular_triplets(Y, k=requested)
            )

            if singular_values[-1] <= tau or requested == min(Y.shape):
                break

            requested = min(
                requested + SVD_INCREMENT,
                min(Y.shape),
            )

        retained = singular_values > tau
        U = U_all[:, retained]
        VT = VT_all[retained, :]
        shrunk_values = singular_values[retained] - tau
        current_rank = int(shrunk_values.size)
        final_backend = backend

        # Evaluate X^k only at observed training entries.
        observed_predictions = predict_low_rank_entries(
            U, shrunk_values, VT, rows, cols
        )
        residual_error = ratings - observed_predictions

        if not np.isfinite(residual_error).all():
            raise FloatingPointError(
                f"Non-finite training residual at original SVT iteration {iteration}."
            )

        # Update Y^k = Y^(k-1) + delta * P_Omega(M - X^k).
        Y.data += delta * residual_error
        previous_rank = current_rank

        if iteration % 50 == 0 or iteration == ORIGINAL_SVT_ITERATIONS:
            print(
                f"Original SVT | iteration {iteration:3d}/"
                f"{ORIGINAL_SVT_ITERATIONS} | rank {current_rank:3d} "
                f"| SVD backend {backend}"
            )

    # Final test prediction: X^K at test coordinates. Test ratings were not
    # used by the iteration, threshold, initialization, or stopping rule.
    test_rows = test_users - 1
    test_cols = test_items - 1
    predictions = predict_low_rank_entries(
        U,
        shrunk_values,
        VT,
        test_rows,
        test_cols,
    )
    predictions = clipped(predictions)

    fit_seconds = time.perf_counter() - start_time
    return predictions, fit_seconds, current_rank, spectral_norm


# ============================================================
# 4. MAIN HOLDOUT EXPERIMENT
# ============================================================

def run_main_experiment() -> pd.DataFrame:
    print("\n" + "=" * 88)
    print("MOVIELENS 100K - FINAL HOLDOUT EXPERIMENT")
    print("=" * 88)
    print("Training set : development split (80%)")
    print("Test set     : untouched final holdout (20%)")
    print("Test usage   : evaluation only; no tuning or early stopping")
    print("\nLocked parameters:")
    print(
        f"Original SVT      : tau={ORIGINAL_SVT_TAU:.1f}, "
        f"delta={ORIGINAL_SVT_DELTA}, iterations={ORIGINAL_SVT_ITERATIONS}"
    )
    print(
        f"Bias-Adjusted SVT : tau multiplier={BIAS_SVT_TAU_MULTIPLIER}, "
        f"delta={BIAS_SVT_DELTA}, iterations={BIAS_SVT_ITERATIONS}"
    )

    development, final_test = load_data()
    test_users, test_items, test_actual = arrays(final_test)

    print("\nData checks:")
    print(f"Development observations : {len(development):,}")
    print(f"Final test observations  : {len(final_test):,}")
    print(f"Development mean rating  : {development['rating'].mean():.6f}")
    print(f"Final test mean rating   : {final_test['rating'].mean():.6f}")

    records = []
    prediction_frame = final_test[
        ["user_id", "item_id", "rating"]
    ].copy()
    prediction_frame = prediction_frame.rename(
        columns={"rating": "actual_rating"}
    )

    # --------------------------------------------------------
    # Model 1: Global Mean
    # --------------------------------------------------------
    print("\n" + "-" * 88)
    print("MODEL 1/4: GLOBAL MEAN")
    start = time.perf_counter()
    global_mean = float(development["rating"].mean())
    global_predictions = np.full(
        len(final_test), global_mean, dtype=np.float64
    )
    elapsed = time.perf_counter() - start

    records.append(
        evaluate(
            "Global Mean",
            global_predictions,
            test_actual,
            elapsed,
            details={"global_mean": global_mean},
        )
    )
    prediction_frame["global_mean_prediction"] = clipped(global_predictions)

    # --------------------------------------------------------
    # Model 2: User-Item Bias
    # --------------------------------------------------------
    print("\n" + "-" * 88)
    print("MODEL 2/4: USER-ITEM BIAS")
    start = time.perf_counter()
    bias_model = UserItemBiasBaseline(
        regularization=BIAS_REGULARIZATION,
        iterations=BIAS_ITERATIONS,
    )
    bias_model.fit(development)
    bias_predictions = bias_model.predict(test_users, test_items)
    elapsed = time.perf_counter() - start

    records.append(
        evaluate(
            "User-Item Bias",
            bias_predictions,
            test_actual,
            elapsed,
            details={
                "regularization": BIAS_REGULARIZATION,
                "bias_iterations": BIAS_ITERATIONS,
            },
        )
    )
    prediction_frame["user_item_bias_prediction"] = clipped(bias_predictions)

    # --------------------------------------------------------
    # Model 3: Original SVT
    # --------------------------------------------------------
    print("\n" + "-" * 88)
    print("MODEL 3/4: ORIGINAL SVT")
    original_predictions, elapsed, original_rank, original_sigma = (
        fit_predict_original_svt(
            development,
            test_users,
            test_items,
        )
    )

    records.append(
        evaluate(
            "Original SVT",
            original_predictions,
            test_actual,
            elapsed,
            rank=original_rank,
            details={
                "tau": ORIGINAL_SVT_TAU,
                "delta": ORIGINAL_SVT_DELTA,
                "iterations": ORIGINAL_SVT_ITERATIONS,
                "training_spectral_norm": original_sigma,
            },
        )
    )
    prediction_frame["original_svt_prediction"] = clipped(original_predictions)

    # --------------------------------------------------------
    # Model 4: Bias-Adjusted SVT (proposed extension)
    # --------------------------------------------------------
    print("\n" + "-" * 88)
    print("MODEL 4/4: BIAS-ADJUSTED SVT")
    start = time.perf_counter()

    bias_svt = BiasAdjustedSVT(
        tau_multiplier=BIAS_SVT_TAU_MULTIPLIER,
        delta=BIAS_SVT_DELTA,
        epsilon=EPSILON,
        max_iterations=BIAS_SVT_ITERATIONS,
        increment=SVD_INCREMENT,
        bias_regularization=BIAS_REGULARIZATION,
        bias_iterations=BIAS_ITERATIONS,
    )
    # No validation frame is supplied: the common iteration (7) was already
    # selected using five-fold development CV. Final test is never involved.
    bias_svt.fit(development, validation=None)
    bias_svt_predictions = bias_svt.predict(
        test_users,
        test_items,
        use_best_validation=False,
    )
    elapsed = time.perf_counter() - start

    records.append(
        evaluate(
            "Bias-Adjusted SVT",
            bias_svt_predictions,
            test_actual,
            elapsed,
            rank=bias_svt.final_rank_,
            details={
                "tau_multiplier": BIAS_SVT_TAU_MULTIPLIER,
                "tau": float(bias_svt.tau),
                "delta": BIAS_SVT_DELTA,
                "iterations": bias_svt.n_iterations_,
                "training_residual_spectral_norm": float(
                    bias_svt.residual_spectral_norm_
                ),
                "k0": bias_svt.k0_,
            },
        )
    )
    prediction_frame["bias_adjusted_svt_prediction"] = clipped(
        bias_svt_predictions
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(records)
    summary_path = TABLES_DIR / "main_experiment_final_test_results.csv"
    predictions_path = TABLES_DIR / "main_experiment_final_test_predictions.csv"
    metadata_path = TABLES_DIR / "main_experiment_configuration.json"

    summary.to_csv(summary_path, index=False)
    prediction_frame.to_csv(predictions_path, index=False)

    metadata = {
        "dataset": "MovieLens 100K",
        "protocol": (
            "80% development split / 20% untouched final test; "
            "five-fold CV performed only within development"
        ),
        "development_observations": int(len(development)),
        "final_test_observations": int(len(final_test)),
        "selected_bias_adjusted_svt": {
            "tau_multiplier": BIAS_SVT_TAU_MULTIPLIER,
            "delta": BIAS_SVT_DELTA,
            "fixed_iterations": BIAS_SVT_ITERATIONS,
        },
        "selected_original_svt": {
            "tau": ORIGINAL_SVT_TAU,
            "delta": ORIGINAL_SVT_DELTA,
            "fixed_iterations": ORIGINAL_SVT_ITERATIONS,
        },
        "bias_baseline": {
            "regularization": BIAS_REGULARIZATION,
            "iterations": BIAS_ITERATIONS,
        },
        "test_set_used_for_model_selection": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("FINAL TEST RESULTS")
    print("=" * 100)
    print(
        summary[
            ["model", "test_rmse", "test_mae", "fit_seconds", "rank"]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    best_index = summary["test_rmse"].idxmin()
    best_row = summary.loc[best_index]
    print(
        f"\nLowest final-test RMSE: {best_row['model']} "
        f"({best_row['test_rmse']:.6f})"
    )

    print("\nSaved:")
    print(summary_path)
    print(predictions_path)
    print(metadata_path)

    return summary


if __name__ == "__main__":
    run_main_experiment()
