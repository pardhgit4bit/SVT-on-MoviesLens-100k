
# ============================================================
# src/bias_adjusted_svt.py
#
# Bias-Adjusted Singular Value Thresholding
#
# Proposed extension:
#   Rating = Global Mean
#          + User Bias
#          + Movie Bias
#          + Low-Rank Residual
#
# SVT is applied to the residual matrix, not raw ratings.
#
# The underlying SVT recurrence and adaptive SVD procedure
# reuse the implementation in src/svt.py.
# ============================================================

from pathlib import Path
import logging
import time

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

from sklearn.metrics import mean_squared_error, mean_absolute_error

from baselines import UserItemBiasBaseline

from svt import (
    compute_leading_singular_triplets,
    predict_low_rank_entries,
    N_USERS,
    N_ITEMS,
    SVD_INCREMENT,
    EPSILON
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Preliminary threshold screening.
# These are multiples of the residual matrix spectral norm.
TAU_MULTIPLIERS = [5, 10, 15, 20]

DELTA = 1.0
MAX_ITERATIONS = 30

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s"
)


# ============================================================
# 2. BIAS-ADJUSTED SVT MODEL
# ============================================================

class BiasAdjustedSVT:
    """
    Bias-adjusted matrix completion using SVT.

    Step 1:
        Fit global, user, and movie biases.

    Step 2:
        Calculate observed residuals.

    Step 3:
        Apply SVT to the sparse residual matrix.

    Step 4:
        Combine bias predictions and low-rank residual
        predictions.
    """

    def __init__(
        self,
        tau_multiplier=10.0,
        delta=DELTA,
        epsilon=EPSILON,
        max_iterations=MAX_ITERATIONS,
        increment=SVD_INCREMENT,
        n_users=N_USERS,
        n_items=N_ITEMS,
        bias_regularization=10.0,
        bias_iterations=20
    ):

        if tau_multiplier <= 0:
            raise ValueError(
                "tau_multiplier must be positive."
            )

        if delta <= 0:
            raise ValueError(
                "delta must be positive."
            )

        if epsilon <= 0:
            raise ValueError(
                "epsilon must be positive."
            )

        if max_iterations < 1:
            raise ValueError(
                "max_iterations must be positive."
            )

        self.tau_multiplier = float(tau_multiplier)
        self.delta = float(delta)
        self.epsilon = float(epsilon)

        self.max_iterations = int(max_iterations)
        self.increment = int(increment)

        self.n_users = int(n_users)
        self.n_items = int(n_items)

        self.bias_regularization = bias_regularization
        self.bias_iterations = bias_iterations

        self.history_ = []

    # ========================================================
    # 2.1. ADAPTIVE SINGULAR VALUE THRESHOLDING
    # ========================================================

    def threshold_sparse_matrix(self, Y, previous_rank):

        min_dimension = min(Y.shape)

        # Start with previous rank + 1.
        s_k = previous_rank + 1

        while True:

            s_k = min(
                s_k,
                min_dimension
            )

            U, singular_values, VT, backend = (
                compute_leading_singular_triplets(
                    Y,
                    k=s_k
                )
            )

            # We have computed enough singular values if
            # the smallest computed one is <= tau.
            if singular_values[-1] <= self.tau:

                break

            # All singular values have been computed.
            if s_k == min_dimension:

                break

            s_k = min(
                s_k + self.increment,
                min_dimension
            )

        # Retain singular values greater than tau.
        retained = singular_values > self.tau

        U = U[:, retained]
        VT = VT[retained, :]

        # Soft-thresholding.
        shrunk_values = (
            singular_values[retained] - self.tau
        )

        current_rank = len(shrunk_values)

        return (
            U,
            shrunk_values,
            VT,
            current_rank,
            s_k,
            backend
        )

    # ========================================================
    # 2.2. FIT
    # ========================================================

    def fit(self, train, validation=None):

        start_time = time.perf_counter()

        # ----------------------------------------------------
        # STEP 1: Prepare training data
        # ----------------------------------------------------

        required_columns = {
            "user_id",
            "item_id",
            "rating"
        }

        if not required_columns.issubset(train.columns):

            raise ValueError(
                "Training data must contain user_id, "
                "item_id, and rating."
            )

        train = (
            train.sort_values(
                ["user_id", "item_id"]
            )
            .reset_index(drop=True)
        )

        if train.duplicated(
            ["user_id", "item_id"]
        ).any():

            raise ValueError(
                "Duplicate user-movie pairs found."
            )

        users = train["user_id"].to_numpy(
            dtype=np.int64
        )

        items = train["item_id"].to_numpy(
            dtype=np.int64
        )

        ratings = train["rating"].to_numpy(
            dtype=np.float64
        )

        rows = users - 1
        cols = items - 1

        if (
            np.any(rows < 0)
            or np.any(rows >= self.n_users)
            or np.any(cols < 0)
            or np.any(cols >= self.n_items)
        ):

            raise ValueError(
                "User or movie ID outside matrix bounds."
            )

        if not np.all(np.isfinite(ratings)):

            raise ValueError(
                "Training ratings contain non-finite values."
            )

        m = len(ratings)

        if m == 0:

            raise ValueError(
                "Training dataset is empty."
            )

        # ----------------------------------------------------
        # STEP 2: Fit user-item bias model
        # ----------------------------------------------------

        self.bias_model_ = UserItemBiasBaseline(
            regularization=self.bias_regularization,
            iterations=self.bias_iterations
        )

        self.bias_model_.fit(train)

        bias_train_predictions = self.bias_model_.predict(
            users,
            items
        )

        # ----------------------------------------------------
        # STEP 3: Calculate observed residuals
        # ----------------------------------------------------

        residuals = (
            ratings - bias_train_predictions
        )

        residual_norm = np.linalg.norm(residuals)

        if residual_norm < 1e-12:

            raise ValueError(
                "Residual matrix has almost zero norm. "
                "SVT cannot be meaningfully applied."
            )

        # Construct sparse residual matrix.
        observed_residual_matrix = csr_matrix(
            (
                residuals,
                (rows, cols)
            ),
            shape=(
                self.n_users,
                self.n_items
            ),
            dtype=np.float64
        )

        # Verify CSR ordering.
        if not np.array_equal(
            observed_residual_matrix.indices,
            cols
        ):

            raise RuntimeError(
                "Residual matrix coordinate ordering mismatch."
            )

        if len(observed_residual_matrix.data) != m:

            raise RuntimeError(
                "Unexpected number of residual observations."
            )

        self.rows_ = rows
        self.cols_ = cols
        self.residuals_ = residuals

        # ----------------------------------------------------
        # STEP 4: Prepare validation data
        # ----------------------------------------------------

        if validation is not None:

            validation = validation.reset_index(
                drop=True
            )

            val_users = validation["user_id"].to_numpy(
                dtype=np.int64
            )

            val_items = validation["item_id"].to_numpy(
                dtype=np.int64
            )

            val_actual = validation["rating"].to_numpy(
                dtype=np.float64
            )

            val_rows = val_users - 1
            val_cols = val_items - 1

            if (
                np.any(val_rows < 0)
                or np.any(val_rows >= self.n_users)
                or np.any(val_cols < 0)
                or np.any(val_cols >= self.n_items)
            ):

                raise ValueError(
                    "Validation IDs outside matrix bounds."
                )

            if not np.all(np.isfinite(val_actual)):

                raise ValueError(
                    "Validation ratings contain non-finite values."
                )

            # Bias predictions for validation observations.
            bias_validation_predictions = (
                self.bias_model_.predict(
                    val_users,
                    val_items
                )
            )

        # ----------------------------------------------------
        # STEP 5: Calculate residual spectral norm
        # ----------------------------------------------------

        _, singular_values, _, backend = (
            compute_leading_singular_triplets(
                observed_residual_matrix,
                k=1
            )
        )

        spectral_norm = singular_values[0]

        if spectral_norm <= 1e-12:

            raise ValueError(
                "Residual matrix has zero spectral norm."
            )

        self.residual_spectral_norm_ = spectral_norm

        # Data-adaptive threshold.
        #
        # tau = multiplier * spectral_norm
        #
        # This is our experimental parameterization,
        # not a threshold rule prescribed by the paper.

        self.tau = (
            self.tau_multiplier * spectral_norm
        )

        # ----------------------------------------------------
        # STEP 6: Initialize Y0 using the paper's strategy
        # ----------------------------------------------------

        p = m / (
            self.n_users * self.n_items
        )

        self.sampling_density_ = p

        # Since tau = multiplier * spectral_norm:
        #
        # k0 = ceil(tau / (delta * spectral_norm))

        ratio = (
            self.tau /
            (self.delta * spectral_norm)
        )

        self.k0_ = max(
            1,
            int(np.ceil(ratio))
        )

        Y = observed_residual_matrix.copy()

        Y.data *= (
            self.k0_ * self.delta
        )

        # ----------------------------------------------------
        # STEP 7: Initialize SVT iterations
        # ----------------------------------------------------

        previous_rank = 0

        denominator = residual_norm

        U = np.empty(
            (self.n_users, 0)
        )

        shrunk_values = np.empty(0)

        VT = np.empty(
            (0, self.n_items)
        )

        self.history_ = []

        self.converged_ = False

        self.best_validation_rmse_ = np.inf
        self.best_validation_mae_ = np.inf

        self.best_validation_iteration_ = None
        self.best_validation_rank_ = None

        self.best_U_ = None
        self.best_singular_values_ = None
        self.best_VT_ = None

        self.last_svd_backend_ = backend

        # ----------------------------------------------------
        # STEP 8: Main SVT iterations
        # ----------------------------------------------------

        for iteration in range(
            1,
            self.max_iterations + 1
        ):

            # -----------------------------------------------
            # X^k = D_tau(Y^(k-1))
            # -----------------------------------------------

            (
                U,
                shrunk_values,
                VT,
                current_rank,
                computed_rank,
                backend
            ) = self.threshold_sparse_matrix(
                Y,
                previous_rank
            )

            # -----------------------------------------------
            # Predict observed residual entries
            # -----------------------------------------------

            observed_residual_predictions = (
                predict_low_rank_entries(
                    U,
                    shrunk_values,
                    VT,
                    rows,
                    cols
                )
            )

            # -----------------------------------------------
            # Calculate observed residual error
            # -----------------------------------------------

            residual_error = (
                residuals -
                observed_residual_predictions
            )

            relative_error = (
                np.linalg.norm(residual_error) /
                denominator
            )

            observed_rmse = np.sqrt(
                np.mean(residual_error ** 2)
            )

            # -----------------------------------------------
            # Validation evaluation
            # -----------------------------------------------

            validation_rmse = np.nan
            validation_mae = np.nan

            if validation is not None:

                val_residual_predictions = (
                    predict_low_rank_entries(
                        U,
                        shrunk_values,
                        VT,
                        val_rows,
                        val_cols
                    )
                )

                # Combine user-item biases and residual estimates.
                combined_predictions = (
                    bias_validation_predictions
                    + val_residual_predictions
                )

                # Clip only the final rating predictions.
                combined_predictions = np.clip(
                    combined_predictions,
                    1,
                    5
                )

                validation_rmse = np.sqrt(
                    mean_squared_error(
                        val_actual,
                        combined_predictions
                    )
                )

                validation_mae = mean_absolute_error(
                    val_actual,
                    combined_predictions
                )

                # Save the best validation model.
                if (
                    validation_rmse <
                    self.best_validation_rmse_
                ):

                    self.best_validation_rmse_ = (
                        validation_rmse
                    )

                    self.best_validation_mae_ = (
                        validation_mae
                    )

                    self.best_validation_iteration_ = (
                        iteration
                    )

                    self.best_validation_rank_ = (
                        current_rank
                    )

                    self.best_U_ = U.copy()

                    self.best_singular_values_ = (
                        shrunk_values.copy()
                    )

                    self.best_VT_ = VT.copy()

            # -----------------------------------------------
            # Save iteration history
            # -----------------------------------------------

            self.history_.append({

                "iteration": iteration,

                "rank": current_rank,

                "computed_singular_values": computed_rank,

                "relative_observed_error": relative_error,

                "observed_residual_rmse": observed_rmse,

                "validation_rmse": validation_rmse,

                "validation_mae": validation_mae,

                "svd_backend": backend

            })

            # -----------------------------------------------
            # Log progress
            # -----------------------------------------------

            if (
                iteration == 1
                or iteration % 25 == 0
            ):

                logging.info(
                    "Iteration %d | Rank: %d | "
                    "Observed residual error: %.6f | "
                    "Validation RMSE: %.4f",
                    iteration,
                    current_rank,
                    relative_error,
                    validation_rmse
                )

            # -----------------------------------------------
            # Paper's observed-entry stopping criterion
            # -----------------------------------------------

            if relative_error <= self.epsilon:

                self.converged_ = True

                logging.info(
                    "Observed residual stopping criterion "
                    "reached at iteration %d.",
                    iteration
                )

                break

            # -----------------------------------------------
            # Update sparse dual matrix
            # -----------------------------------------------

            Y.data += (
                self.delta * residual_error
            )

            previous_rank = current_rank

        # ----------------------------------------------------
        # STEP 9: Save final model
        # ----------------------------------------------------

        self.execution_time_ = (
            time.perf_counter() - start_time
        )

        self.U_ = U
        self.shrunk_values_ = shrunk_values
        self.VT_ = VT

        self.final_rank_ = current_rank
        self.n_iterations_ = len(self.history_)

        self.final_relative_error_ = relative_error

        self.final_observed_rmse_ = observed_rmse

        self.last_svd_backend_ = backend

        if not self.converged_:

            logging.warning(
                "Maximum iterations reached before the "
                "observed residual tolerance was satisfied."
            )

        return self

    # ========================================================
    # 2.3. PREDICTION
    # ========================================================

    def predict(
        self,
        user_ids,
        item_ids,
        use_best_validation=True
    ):
        """
        Predict ratings using:

            bias prediction + low-rank residual prediction.

        The best validation iteration is used by default.
        """

        user_ids = np.asarray(
            user_ids,
            dtype=np.int64
        )

        item_ids = np.asarray(
            item_ids,
            dtype=np.int64
        )

        if len(user_ids) != len(item_ids):

            raise ValueError(
                "User and item ID arrays must have equal lengths."
            )

        rows = user_ids - 1
        cols = item_ids - 1

        if (
            np.any(rows < 0)
            or np.any(rows >= self.n_users)
            or np.any(cols < 0)
            or np.any(cols >= self.n_items)
        ):

            raise ValueError(
                "User or movie ID outside matrix bounds."
            )

        # Bias prediction.
        bias_predictions = self.bias_model_.predict(
            user_ids,
            item_ids
        )

        # Select final or best validation factors.
        if use_best_validation:

            if self.best_U_ is None:

                raise ValueError(
                    "Best validation model is unavailable."
                )

            U = self.best_U_

            singular_values = (
                self.best_singular_values_
            )

            VT = self.best_VT_

        else:

            U = self.U_

            singular_values = self.shrunk_values_

            VT = self.VT_

        # Low-rank residual prediction.
        residual_predictions = predict_low_rank_entries(
            U,
            singular_values,
            VT,
            rows,
            cols
        )

        # Combine both predictions.
        predictions = (
            bias_predictions +
            residual_predictions
        )

        return np.clip(
            predictions,
            1,
            5
        )


# ============================================================
# 3. FIVE-FOLD CROSS-VALIDATION
# ============================================================

def run_cross_validation():

    print("\n" + "=" * 85)
    print("BIAS-ADJUSTED SVT - FIVE-FOLD CROSS-VALIDATION")
    print("=" * 85)

    fold_records = []

    for tau_multiplier in TAU_MULTIPLIERS:

        print("\n" + "#" * 85)
        print(f"TAU MULTIPLIER: {tau_multiplier}")
        print("#" * 85)

        for fold in range(1, 6):

            print("\n" + "-" * 75)
            print(
                f"FOLD {fold}/5 | "
                f"Tau multiplier: {tau_multiplier}"
            )
            print("-" * 75)

            train_path = SPLIT_DIR / f"fold_{fold}_train.csv"
            validation_path = (
                SPLIT_DIR / f"fold_{fold}_validation.csv"
            )

            if not train_path.exists() or not validation_path.exists():
                raise FileNotFoundError(
                    f"Missing split files for fold {fold}. "
                    "Run src/preprocessing.py first."
                )

            train = pd.read_csv(train_path)
            validation = pd.read_csv(validation_path)

            model = BiasAdjustedSVT(
                tau_multiplier=tau_multiplier,
                delta=DELTA,
                epsilon=EPSILON,
                max_iterations=MAX_ITERATIONS,
                increment=SVD_INCREMENT
            )

            model.fit(train, validation=validation)

            # Re-evaluate the selected best-validation iterate.
            predictions = model.predict(
                validation["user_id"].to_numpy(),
                validation["item_id"].to_numpy(),
                use_best_validation=True
            )

            actual = validation["rating"].to_numpy(dtype=np.float64)

            best_rmse = np.sqrt(
                mean_squared_error(actual, predictions)
            )
            best_mae = mean_absolute_error(actual, predictions)

            # Save the complete iteration history for this fold.
            history = pd.DataFrame(model.history_)
            history.insert(0, "fold", fold)
            history.insert(1, "tau_multiplier", tau_multiplier)

            history_path = (
                TABLES_DIR
                / f"bias_adjusted_svt_tau_{tau_multiplier}_fold_{fold}_history.csv"
            )
            history.to_csv(history_path, index=False)

            record = {
                "fold": fold,
                "tau_multiplier": tau_multiplier,
                "tau": model.tau,
                "residual_spectral_norm": model.residual_spectral_norm_,
                "delta": model.delta,
                "k0": model.k0_,
                "best_iteration": model.best_validation_iteration_,
                "best_rank": model.best_validation_rank_,
                "best_validation_rmse": best_rmse,
                "best_validation_mae": best_mae,
                "final_rank": model.final_rank_,
                "final_observed_error": model.final_relative_error_,
                "iterations": model.n_iterations_,
                "converged": model.converged_,
                "execution_time_seconds": model.execution_time_,
                "svd_backend": model.last_svd_backend_,
            }

            fold_records.append(record)

            # Save after every fold so completed work is not lost.
            fold_results = pd.DataFrame(fold_records)
            fold_results.to_csv(
                TABLES_DIR / "bias_adjusted_svt_cv_fold_results.csv",
                index=False
            )

            print(f"Residual spectral norm : {model.residual_spectral_norm_:.4f}")
            print(f"Actual tau             : {model.tau:.4f}")
            print(f"k0                     : {model.k0_}")
            print(f"Best iteration         : {model.best_validation_iteration_}")
            print(f"Best rank              : {model.best_validation_rank_}")
            print(f"Validation RMSE        : {best_rmse:.4f}")
            print(f"Validation MAE         : {best_mae:.4f}")
            print(f"Final observed error   : {model.final_relative_error_:.6f}")
            print(f"Execution time         : {model.execution_time_:.2f} seconds")

    # --------------------------------------------------------
    # Aggregate fold-level results for each tau multiplier.
    # --------------------------------------------------------

    fold_results = pd.DataFrame(fold_records)

    summary = (
        fold_results
        .groupby("tau_multiplier")
        .agg(
            mean_validation_rmse=("best_validation_rmse", "mean"),
            std_validation_rmse=("best_validation_rmse", "std"),
            mean_validation_mae=("best_validation_mae", "mean"),
            std_validation_mae=("best_validation_mae", "std"),
            mean_best_iteration=("best_iteration", "mean"),
            mean_best_rank=("best_rank", "mean"),
            mean_final_rank=("final_rank", "mean"),
            mean_execution_time_seconds=("execution_time_seconds", "mean"),
            total_execution_time_seconds=("execution_time_seconds", "sum"),
            converged_folds=("converged", "sum"),
            number_of_folds=("fold", "count"),
        )
        .reset_index()
    )

    summary.to_csv(
        TABLES_DIR / "bias_adjusted_svt_cv_summary.csv",
        index=False
    )

    print("\n" + "=" * 100)
    print("BIAS-ADJUSTED SVT - FOLD RESULTS")
    print("=" * 100)
    print(
        fold_results.to_string(
            index=False,
            float_format="%.4f"
        )
    )

    print("\n" + "=" * 100)
    print("BIAS-ADJUSTED SVT - FIVE-FOLD SUMMARY")
    print("=" * 100)
    print(
        summary.to_string(
            index=False,
            float_format="%.4f"
        )
    )

    print("\nResults saved to:")
    print(TABLES_DIR / "bias_adjusted_svt_cv_fold_results.csv")
    print(TABLES_DIR / "bias_adjusted_svt_cv_summary.csv")
    print(
        "Per-fold iteration histories: "
        f"{TABLES_DIR / 'bias_adjusted_svt_tau_*_fold_*_history.csv'}"
    )

    return fold_results, summary


if __name__ == "__main__":
    run_cross_validation()
