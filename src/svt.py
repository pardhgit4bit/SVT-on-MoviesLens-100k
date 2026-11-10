
# ============================================================
# src/svt.py
#
# Singular Value Thresholding for MovieLens 100K
#
# Based on:
# Cai, Candès, and Shen (2010)
# "A Singular Value Thresholding Algorithm for Matrix Completion"
#
# Implements Algorithm 1 with:
#   - Sparse dual matrix Y
#   - Adaptive partial SVD
#   - Singular value soft-thresholding
#   - Paper-based initialization
#   - Observed-entry reconstruction stopping criterion
#
# Experimental additions:
#   - Validation RMSE and MAE monitoring
#   - Best validation iteration tracking
#   - Iteration history export
# ============================================================


from pathlib import Path
import logging
import time

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds, ArpackNoConvergence

from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)


# MovieLens 100K dimensions
N_USERS = 943
N_ITEMS = 1682


# Initial threshold.
# This is an experimental starting value, not a universal
# recommendation from the paper.
TAU_INITIAL = 5.0 * min(N_USERS, N_ITEMS)


# SVT parameters
DELTA_INITIAL = 1.0
EPSILON = 1e-4

MAX_ITERATIONS = 500
SVD_INCREMENT = 5

RANDOM_STATE = 42


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


# ============================================================
# 2. PARTIAL SVD
# ============================================================


def compute_leading_singular_triplets(matrix, k):

    min_dimension = min(matrix.shape)

    if not 1 <= k <= min_dimension:
        raise ValueError(
            "Invalid number of singular values requested."
        )

    if matrix.nnz == 0 or not np.any(matrix.data):

        return (
            np.zeros((matrix.shape[0], k)),
            np.zeros(k),
            np.zeros((k, matrix.shape[1])),
            "ZERO-MATRIX"
        )

    if k == min_dimension:

        U, singular_values, VT = np.linalg.svd(
            matrix.toarray(),
            full_matrices=False
        )

        return U, singular_values, VT, "DENSE-FULL-SVD"

    # Reproducible starting vector.
    rng = np.random.default_rng(RANDOM_STATE)

    v0 = rng.standard_normal(matrix.shape[0])
    v0 /= np.linalg.norm(v0)

    # Give PROPACK more iterations than its default 10*k.
    propack_maxiter = max(200, 20 * k)

    try:

        U, singular_values, VT = svds(
            matrix,
            k=k,
            which="LM",
            solver="propack",
            tol=1e-6,
            maxiter=propack_maxiter,
            v0=v0
        )

        backend = "PROPACK"

    except (
        TypeError,
        ValueError,
        RuntimeError,
        np.linalg.LinAlgError,
        ArpackNoConvergence
    ) as error:

        logging.warning(
            "PROPACK failed: %s. Falling back to ARPACK.",
            error
        )

        # ARPACK requires a starting vector of length
        # min(matrix.shape).
        rng = np.random.default_rng(RANDOM_STATE)

        arpack_v0 = rng.standard_normal(min_dimension)
        arpack_v0 /= np.linalg.norm(arpack_v0)

        U, singular_values, VT = svds(
            matrix,
            k=k,
            which="LM",
            solver="arpack",
            tol=1e-6,
            maxiter=10 * min_dimension,
            v0=arpack_v0
        )

        backend = "ARPACK"

    # Return singular values in descending order.
    order = np.argsort(singular_values)[::-1]

    U = U[:, order]
    singular_values = singular_values[order]
    VT = VT[order, :]

    return U, singular_values, VT, backend



# ============================================================
# 3. LOW-RANK MATRIX OPERATIONS
# ============================================================

def predict_low_rank_entries(
    U,
    singular_values,
    VT,
    rows,
    cols
):
    """
    Calculate selected entries of:

        X = U @ diag(singular_values) @ VT

    without constructing the complete dense matrix X.

    This preserves the low-rank representation used by SVT.
    """

    if len(singular_values) == 0:

        return np.zeros(
            len(rows),
            dtype=np.float64
        )

    predictions = np.sum(
        U[rows, :]
        * singular_values[np.newaxis, :]
        * VT[:, cols].T,
        axis=1
    )

    return predictions


# ============================================================
# 4. SINGULAR VALUE THRESHOLDING MODEL
# ============================================================

class SingularValueThresholding:
    """
    Paper-based Singular Value Thresholding algorithm.

    The primary recurrence is:

        X^k = D_tau(Y^(k-1))

        Y^k = Y^(k-1) + delta * P_Omega(M - X^k)

    Only observed entries are updated in Y.

    X is maintained in low-rank SVD form.
    """

    def __init__(
        self,
        tau=TAU_INITIAL,
        delta=DELTA_INITIAL,
        epsilon=EPSILON,
        max_iterations=MAX_ITERATIONS,
        increment=SVD_INCREMENT,
        n_users=N_USERS,
        n_items=N_ITEMS
    ):

        if tau <= 0:
            raise ValueError("tau must be positive.")

        if delta is not None and delta <= 0:
            raise ValueError("delta must be positive.")

        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        if max_iterations < 1:
            raise ValueError(
                "max_iterations must be positive."
            )

        if increment < 1:
            raise ValueError(
                "SVD increment must be positive."
            )

        self.tau = float(tau)
        self.delta = delta

        self.epsilon = float(epsilon)
        self.max_iterations = int(max_iterations)
        self.increment = int(increment)

        self.n_users = int(n_users)
        self.n_items = int(n_items)

        self.history_ = []

    # ========================================================
    # 4.1. INITIALIZE DUAL MATRIX
    # ========================================================

    def initialize_dual_matrix(
        self,
        observed_matrix,
        number_of_observations
    ):
        """
        Implements paper Section 5.1.2 and 5.1.3.

        Calculate:

            p = m / (n1 * n2)

        Paper's empirical step-size rule:

            delta = 1.2 / p

        Initialization:

            k0 = ceil(tau / (delta * ||P_Omega(M)||_2))

            Y0 = k0 * delta * P_Omega(M)

        The main experiment explicitly uses delta=1.0,
        which satisfies the general sufficient condition
        0 < delta < 2 from the paper's convergence theorem.
        """

        n1 = self.n_users
        n2 = self.n_items

        m = number_of_observations

        # Sampling density
        p = m / (n1 * n2)

        self.sampling_density_ = p

        # Choose step size.
        if self.delta is None:

            self.delta_ = 1.2 / p

        else:

            self.delta_ = float(self.delta)

        if self.delta_ >= 2:

            logging.warning(
                "delta=%.4f is >= 2. The general "
                "convergence theorem requires delta < 2. "
                "The paper's larger 1.2/p rule is an "
                "empirical heuristic under additional "
                "sampling assumptions.",
                self.delta_
            )

        # Calculate spectral norm of observed matrix.
        _, singular_values, _, backend = (
            compute_leading_singular_triplets(
                observed_matrix,
                k=1
            )
        )

        spectral_norm = singular_values[0]

        if spectral_norm <= 0:

            raise ValueError(
                "Observed matrix has zero spectral norm."
            )

        self.spectral_norm_ = spectral_norm

        self.svd_backend_initialization_ = backend

        # Paper equation (5.3).
        ratio = (
            self.tau /
            (self.delta_ * spectral_norm)
        )

        self.k0_ = max(
            1,
            int(np.ceil(ratio))
        )

        # Paper Algorithm 1, initialization.
        Y = observed_matrix.copy()

        Y.data *= (
            self.k0_ * self.delta_
        )

        return Y

    # ========================================================
    # 4.2. ADAPTIVE SINGULAR VALUE THRESHOLDING
    # ========================================================

    def threshold_sparse_matrix(
        self,
        Y,
        previous_rank
    ):
        """
        Implements Algorithm 1, lines 4-10.

        Start with:

            s_k = r_(k-1) + 1

        If the smallest computed singular value is above tau,
        increase s_k by the predefined increment.

        Retain only singular values greater than tau.
        """

        min_dimension = min(Y.shape)

        # Algorithm 1, line 4.
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

            # If the smallest computed singular value
            # is below or equal to tau, enough values
            # have been computed.
            if singular_values[-1] <= self.tau:

                break

            # All singular values have been computed.
            if s_k == min_dimension:

                break

            # Algorithm 1, line 7.
            s_k = min(
                s_k + self.increment,
                min_dimension
            )

        # Algorithm 1, line 9.
        retained = singular_values > self.tau

        U = U[:, retained]
        VT = VT[retained, :]

        # Algorithm 1, line 10.
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
    # 4.3. MODEL FITTING
    # ========================================================

    def fit(
        self,
        train,
        validation=None
    ):
        """
        Fit SVT using training observations.

        Validation observations are only used to monitor
        generalization performance. They are NEVER used
        in the dual-matrix update.
        """

        start_time = time.perf_counter()

        logging.info(
            "Preparing observed rating matrix..."
        )

        required_columns = {
            "user_id",
            "item_id",
            "rating"
        }

        if not required_columns.issubset(train.columns):

            raise ValueError(
                f"Training data must contain {required_columns}."
            )

        # Sort data so the CSR data ordering matches
        # the observed-coordinate ordering.
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

        # Check matrix dimensions.
        if (
            np.any(rows < 0)
            or np.any(rows >= self.n_users)
            or np.any(cols < 0)
            or np.any(cols >= self.n_items)
        ):

            raise ValueError(
                "User or movie ID is outside matrix bounds."
            )

        if np.any((ratings < 1) | (ratings > 5)):

            raise ValueError(
                "Ratings must lie between 1 and 5."
            )

        m = len(ratings)

        if m == 0:

            raise ValueError(
                "Training dataset is empty."
            )

        # ----------------------------------------------------
        # Construct P_Omega(M)
        # ----------------------------------------------------

        observed_matrix = csr_matrix(
            (
                ratings,
                (rows, cols)
            ),
            shape=(
                self.n_users,
                self.n_items
            ),
            dtype=np.float64
        )

        # The sparse matrix data must have the same
        # observation ordering as rows, cols, and ratings.
        if not np.array_equal(
            observed_matrix.indices,
            cols
        ):

            raise RuntimeError(
                "CSR observation ordering does not match "
                "the training coordinates."
            )

        if len(observed_matrix.data) != m:

            raise RuntimeError(
                "Unexpected number of sparse observations."
            )

        self.rows_ = rows
        self.cols_ = cols
        self.ratings_ = ratings

        # ----------------------------------------------------
        # Prepare validation data
        # ----------------------------------------------------

        if validation is not None:

            validation_columns = {
                "user_id",
                "item_id",
                "rating"
            }

            if not validation_columns.issubset(
                validation.columns
            ):

                raise ValueError(
                    "Validation data must contain user_id, "
                    "item_id, and rating."
                )

            validation = validation.reset_index(
                drop=True
            )

            val_rows = (
                validation["user_id"].to_numpy(
                    dtype=np.int64
                ) - 1
            )

            val_cols = (
                validation["item_id"].to_numpy(
                    dtype=np.int64
                ) - 1
            )

            val_actual = validation["rating"].to_numpy(
                dtype=np.float64
            )

            if len(val_actual) == 0:

                raise ValueError(
                    "Validation dataset is empty."
                )

            if (
                np.any(val_rows < 0)
                or np.any(val_rows >= self.n_users)
                or np.any(val_cols < 0)
                or np.any(val_cols >= self.n_items)
            ):

                raise ValueError(
                    "Validation IDs are outside matrix bounds."
                )

            if np.any(
                (val_actual < 1) | (val_actual > 5)
            ):

                raise ValueError(
                    "Validation ratings must lie between 1 and 5."
                )

        # ----------------------------------------------------
        # Initialize Y0
        # ----------------------------------------------------

        logging.info(
            "Initializing dual matrix Y0..."
        )

        Y = self.initialize_dual_matrix(
            observed_matrix,
            m
        )

        logging.info(
            "Matrix shape: %d x %d | Observations: %d",
            self.n_users,
            self.n_items,
            m
        )

        logging.info(
            "Sampling density: %.6f | delta: %.6f | tau: %.2f",
            self.sampling_density_,
            self.delta_,
            self.tau
        )

        logging.info(
            "Initial k0: %d | Initial SVD backend: %s",
            self.k0_,
            self.svd_backend_initialization_
        )

        # ----------------------------------------------------
        # Initialize the SVT iteration
        # ----------------------------------------------------

        # Algorithm 1, line 2.
        previous_rank = 0

        denominator = np.linalg.norm(ratings)

        if denominator <= 0:

            raise ValueError(
                "Observed ratings have zero norm."
            )

        # Initial empty low-rank representation.
        U = np.empty(
            (self.n_users, 0)
        )

        shrunk_values = np.empty(0)

        VT = np.empty(
            (0, self.n_items)
        )

        self.history_ = []

        self.converged_ = False

        # Best validation model tracking.
        self.best_validation_rmse_ = np.inf
        self.best_validation_mae_ = np.inf
        self.best_validation_iteration_ = None
        self.best_validation_rank_ = None

        self.best_U_ = None
        self.best_singular_values_ = None
        self.best_VT_ = None

        logging.info(
            "Starting SVT iterations..."
        )

        # ----------------------------------------------------
        # Main SVT loop
        # ----------------------------------------------------

        for iteration in range(
            1,
            self.max_iterations + 1
        ):

            # =================================================
            # Step 1: Compute X^k = D_tau(Y^(k-1))
            # =================================================

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

            # =================================================
            # Step 2: Compute observed-entry predictions
            # =================================================

            observed_predictions = predict_low_rank_entries(
                U,
                shrunk_values,
                VT,
                rows,
                cols
            )

            # =================================================
            # Step 3: Calculate observed reconstruction error
            # =================================================

            residual = (
                ratings - observed_predictions
            )

            relative_error = (
                np.linalg.norm(residual) /
                denominator
            )

            observed_rmse = np.sqrt(
                np.mean(residual ** 2)
            )

            # =================================================
            # Step 4: Monitor validation performance
            # =================================================

            validation_rmse = np.nan
            validation_mae = np.nan

            if validation is not None:

                val_predictions = predict_low_rank_entries(
                    U,
                    shrunk_values,
                    VT,
                    val_rows,
                    val_cols
                )

                # Prediction clipping is evaluation
                # post-processing, not part of SVT.
                val_predictions = np.clip(
                    val_predictions,
                    1,
                    5
                )

                validation_rmse = np.sqrt(
                    mean_squared_error(
                        val_actual,
                        val_predictions
                    )
                )

                validation_mae = mean_absolute_error(
                    val_actual,
                    val_predictions
                )

                # Track best validation iteration.
                if (
                    validation_rmse
                    < self.best_validation_rmse_
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

                    # Preserve the corresponding low-rank factors.
                    self.best_U_ = U.copy()

                    self.best_singular_values_ = (
                        shrunk_values.copy()
                    )

                    self.best_VT_ = VT.copy()

            # =================================================
            # Step 5: Save iteration history
            # =================================================

            self.history_.append({

                "iteration": iteration,

                "rank": current_rank,

                "computed_singular_values": computed_rank,

                "relative_observed_error": relative_error,

                "observed_rmse": observed_rmse,

                "validation_rmse": validation_rmse,

                "validation_mae": validation_mae,

                "svd_backend": backend

            })

            # =================================================
            # Step 6: Logging
            # =================================================

            if (
                iteration == 1
                or iteration % 25 == 0
            ):

                if validation is not None:

                    logging.info(
                        "Iteration %d | Rank: %d | "
                        "Observed error: %.6f | "
                        "Validation RMSE: %.4f",
                        iteration,
                        current_rank,
                        relative_error,
                        validation_rmse
                    )

                else:

                    logging.info(
                        "Iteration %d | Rank: %d | "
                        "Observed error: %.6f",
                        iteration,
                        current_rank,
                        relative_error
                    )

            # =================================================
            # Step 7: Paper's stopping criterion
            # =================================================

            if relative_error <= self.epsilon:

                self.converged_ = True

                logging.info(
                    "Paper's stopping criterion reached "
                    "at iteration %d.",
                    iteration
                )

                # X^k is already computed.
                # No further Y update is necessary.
                break

            # =================================================
            # Step 8: Update Y^k
            # =================================================

            # Algorithm 1, line 12:
            #
            # Y^k_ij = Y^(k-1)_ij
            #          + delta * (M_ij - X^k_ij)
            #
            # Only entries belonging to Omega are updated.
            #
            # Since the CSR structure and training coordinates
            # have identical ordering, Y.data is aligned with
            # the residual array.

            Y.data += (
                self.delta_ * residual
            )

            # Prepare for the next iteration.
            previous_rank = current_rank

        # ----------------------------------------------------
        # End of SVT iterations
        # ----------------------------------------------------

        self.execution_time_ = (
            time.perf_counter() - start_time
        )

        # Store final low-rank representation.
        self.U_ = U
        self.shrunk_values_ = shrunk_values
        self.VT_ = VT

        self.final_rank_ = current_rank

        self.n_iterations_ = len(
            self.history_
        )

        self.final_relative_error_ = (
            relative_error
        )

        self.final_observed_rmse_ = (
            observed_rmse
        )

        self.last_svd_backend_ = backend

        if not self.converged_:

            logging.warning(
                "Maximum iterations reached before "
                "the paper's relative observed-error "
                "tolerance was satisfied."
            )

        return self

    # ========================================================
    # 4.4. PREDICTION
    # ========================================================

    def predict(
        self,
        user_ids,
        item_ids,
        use_best_validation=False
    ):
        """
        Predict selected user-movie ratings.

        By default, predictions come from the final iteration.

        Set use_best_validation=True to use the low-rank
        factors saved at the best validation iteration.
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
                "user_ids and item_ids must have equal lengths."
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
                "User or movie ID is outside matrix bounds."
            )

        if use_best_validation:

            if self.best_U_ is None:

                raise ValueError(
                    "No best validation model is available. "
                    "Fit with validation data first."
                )

            U = self.best_U_

            singular_values = (
                self.best_singular_values_
            )

            VT = self.best_VT_

        else:

            U = self.U_

            singular_values = (
                self.shrunk_values_
            )

            VT = self.VT_

        return predict_low_rank_entries(
            U,
            singular_values,
            VT,
            rows,
            cols
        )


# ============================================================
# 5. SINGLE-FOLD EXPERIMENT
# ============================================================

def run_single_fold_experiment():
    """
    First diagnostic experiment on fold 1.

    Reports:
        - Final iteration validation metrics
        - Best validation iteration and its metrics
        - Matrix rank
        - Relative observed reconstruction error
        - Convergence status
        - Execution time
    """

    logging.info(
        "Loading fold 1..."
    )

    train = pd.read_csv(
        SPLIT_DIR / "fold_1_train.csv"
    )

    validation = pd.read_csv(
        SPLIT_DIR / "fold_1_validation.csv"
    )

    # --------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------

    model = SingularValueThresholding(

        tau=TAU_INITIAL,

        # Stable step size satisfying the general
        # sufficient condition 0 < delta < 2.
        #
        # We will investigate alternative values
        # during hyperparameter experiments.
        delta=DELTA_INITIAL,

        epsilon=EPSILON,

        max_iterations=MAX_ITERATIONS,

        increment=SVD_INCREMENT,

        n_users=N_USERS,

        n_items=N_ITEMS

    )

    # --------------------------------------------------------
    # Fit with validation monitoring
    # --------------------------------------------------------

    model.fit(
        train,
        validation=validation
    )

    # --------------------------------------------------------
    # Calculate final-iteration validation metrics
    # --------------------------------------------------------

    final_predictions = model.predict(
        validation["user_id"].to_numpy(),
        validation["item_id"].to_numpy()
    )

    final_predictions = np.clip(
        final_predictions,
        1,
        5
    )

    actual = validation["rating"].to_numpy()

    final_validation_rmse = np.sqrt(
        mean_squared_error(
            actual,
            final_predictions
        )
    )

    final_validation_mae = mean_absolute_error(
        actual,
        final_predictions
    )

    # --------------------------------------------------------
    # Obtain best validation iteration
    # --------------------------------------------------------

    if model.best_validation_iteration_ is not None:

        best_predictions = model.predict(
            validation["user_id"].to_numpy(),
            validation["item_id"].to_numpy(),
            use_best_validation=True
        )

        best_predictions = np.clip(
            best_predictions,
            1,
            5
        )

        best_rmse = np.sqrt(
            mean_squared_error(
                actual,
                best_predictions
            )
        )

        best_mae = mean_absolute_error(
            actual,
            best_predictions
        )

    else:

        best_rmse = np.nan
        best_mae = np.nan

    # --------------------------------------------------------
    # Save iteration history
    # --------------------------------------------------------

    history = pd.DataFrame(
        model.history_
    )

    history_path = (
        TABLES_DIR / "svt_fold1_history.csv"
    )

    history.to_csv(
        history_path,
        index=False
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    print(
        "PAPER-FAITHFUL SVT - FOLD 1"
    )

    print("=" * 65)

    print("\nFINAL ITERATION RESULTS")

    print(
        f"Validation RMSE          : "
        f"{final_validation_rmse:.4f}"
    )

    print(
        f"Validation MAE           : "
        f"{final_validation_mae:.4f}"
    )

    print(
        f"Final matrix rank        : "
        f"{model.final_rank_}"
    )

    print(
        f"Iterations               : "
        f"{model.n_iterations_}"
    )

    print(
        f"Relative observed error  : "
        f"{model.final_relative_error_:.7f}"
    )

    print(
        f"Converged                : "
        f"{model.converged_}"
    )

    print("\nBEST VALIDATION ITERATION")

    print(
        f"Iteration                : "
        f"{model.best_validation_iteration_}"
    )

    print(
        f"Validation RMSE          : "
        f"{best_rmse:.4f}"
    )

    print(
        f"Validation MAE           : "
        f"{best_mae:.4f}"
    )

    print(
        f"Matrix rank              : "
        f"{model.best_validation_rank_}"
    )

    print("\nEXPERIMENT CONFIGURATION")

    print(
        f"Tau                      : "
        f"{model.tau:.2f}"
    )

    print(
        f"Delta                    : "
        f"{model.delta_:.4f}"
    )

    print(
        f"Initial k0               : "
        f"{model.k0_}"
    )

    print(
        f"SVD backend              : "
        f"{model.last_svd_backend_}"
    )

    print(
        f"Execution time           : "
        f"{model.execution_time_:.2f} seconds"
    )

    print("\nIteration history saved to:")

    print(history_path)

    print("\n" + "=" * 65)


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":

    run_single_fold_experiment()
