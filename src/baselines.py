
# src/baselines.py

from pathlib import Path
import logging
import time

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
BIAS_REGULARIZATION = 10.0
BIAS_ITERATIONS = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


class GlobalMeanBaseline:
    """Predict every rating using the training-set global mean."""

    def fit(self, train):
        self.global_mean = train["rating"].mean()
        return self

    def predict(self, user_ids, item_ids):
        return np.full(
            len(user_ids),
            self.global_mean,
            dtype=np.float64
        )


class UserItemBiasBaseline:
    """Global mean + regularized user and item biases."""

    def __init__(self, regularization=10.0, iterations=20):
        self.regularization = regularization
        self.iterations = iterations

    def fit(self, train):
        users = train["user_id"].to_numpy(dtype=np.int64)
        items = train["item_id"].to_numpy(dtype=np.int64)
        ratings = train["rating"].to_numpy(dtype=np.float64)

        self.global_mean = ratings.mean()

        self.user_ids_, user_idx = np.unique(
            users, return_inverse=True
        )
        self.item_ids_, item_idx = np.unique(
            items, return_inverse=True
        )

        n_users = len(self.user_ids_)
        n_items = len(self.item_ids_)

        user_counts = np.bincount(user_idx, minlength=n_users)
        item_counts = np.bincount(item_idx, minlength=n_items)

        self.user_bias_ = np.zeros(n_users)
        self.item_bias_ = np.zeros(n_items)

        for _ in range(self.iterations):

            # Update user biases
            user_residuals = (
                ratings
                - self.global_mean
                - self.item_bias_[item_idx]
            )

            user_sums = np.bincount(
                user_idx,
                weights=user_residuals,
                minlength=n_users
            )

            self.user_bias_ = user_sums / (
                self.regularization + user_counts
            )

            # Update item biases
            item_residuals = (
                ratings
                - self.global_mean
                - self.user_bias_[user_idx]
            )

            item_sums = np.bincount(
                item_idx,
                weights=item_residuals,
                minlength=n_items
            )

            self.item_bias_ = item_sums / (
                self.regularization + item_counts
            )

        return self

    def predict(self, user_ids, item_ids):
        user_ids = np.asarray(user_ids, dtype=np.int64)
        item_ids = np.asarray(item_ids, dtype=np.int64)

        user_pos = np.searchsorted(self.user_ids_, user_ids)
        item_pos = np.searchsorted(self.item_ids_, item_ids)

        # Unseen users/items receive zero additional bias
        user_bias = np.zeros(len(user_ids))
        item_bias = np.zeros(len(item_ids))

        valid_users = (
            (user_pos < len(self.user_ids_))
            & (self.user_ids_[np.minimum(
                user_pos, len(self.user_ids_) - 1
            )] == user_ids)
        )

        valid_items = (
            (item_pos < len(self.item_ids_))
            & (self.item_ids_[np.minimum(
                item_pos, len(self.item_ids_) - 1
            )] == item_ids)
        )

        user_bias[valid_users] = self.user_bias_[user_pos[valid_users]]
        item_bias[valid_items] = self.item_bias_[item_pos[valid_items]]

        return self.global_mean + user_bias + item_bias


def evaluate_model(model, train, validation):
    """Fit a model and calculate validation metrics."""

    start_time = time.perf_counter()

    model.fit(train)

    predictions = model.predict(
        validation["user_id"].to_numpy(),
        validation["item_id"].to_numpy()
    )

    actual = validation["rating"].to_numpy()

    # Clip predictions to the known rating scale
    predictions = np.clip(predictions, 1, 5)

    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)

    elapsed = time.perf_counter() - start_time

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Time_seconds": elapsed
    }


def run_cross_validation():
    """Evaluate baselines across all five folds."""

    records = []

    for fold in range(1, N_FOLDS + 1):

        logging.info("Starting fold %d...", fold)

        train = pd.read_csv(
            SPLIT_DIR / f"fold_{fold}_train.csv"
        )

        validation = pd.read_csv(
            SPLIT_DIR / f"fold_{fold}_validation.csv"
        )

        models = {
            "Global Mean": GlobalMeanBaseline(),
            "User-Item Bias": UserItemBiasBaseline(
                regularization=BIAS_REGULARIZATION,
                iterations=BIAS_ITERATIONS
            )
        }

        for model_name, model in models.items():

            metrics = evaluate_model(
                model,
                train,
                validation
            )

            records.append({
                "Fold": fold,
                "Model": model_name,
                **metrics
            })

            logging.info(
                "%s | RMSE: %.4f | MAE: %.4f",
                model_name,
                metrics["RMSE"],
                metrics["MAE"]
            )

    results = pd.DataFrame(records)

    # Save individual fold results
    results.to_csv(
        TABLES_DIR / "baseline_fold_results.csv",
        index=False
    )

    # Calculate mean and standard deviation across folds
    summary = (
        results.groupby("Model")
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            Time_mean=("Time_seconds", "mean")
        )
        .reset_index()
    )

    summary.to_csv(
        TABLES_DIR / "baseline_summary.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("BASELINE CROSS-VALIDATION RESULTS")
    print("=" * 65)

    print(summary.to_string(index=False, float_format="%.4f"))

    print("\nResults saved in:", TABLES_DIR)


if __name__ == "__main__":
    run_cross_validation()
