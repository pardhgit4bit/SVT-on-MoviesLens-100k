# src/preprocessing.py

from pathlib import Path
import logging

import pandas as pd
from sklearn.model_selection import train_test_split, KFold

from data_loader import load_ratings, validate_ratings


# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

SPLIT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


def create_holdout_split(df):
    """Create development and untouched final test sets."""

    development, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    development = development.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return development, test


def create_cv_folds(development):
    """Create reproducible 5-fold cross-validation splits."""

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(development), start=1
    ):
        train_data = development.iloc[train_idx].reset_index(drop=True)
        val_data = development.iloc[val_idx].reset_index(drop=True)

        train_data.to_csv(
            SPLIT_DIR / f"fold_{fold}_train.csv",
            index=False
        )

        val_data.to_csv(
            SPLIT_DIR / f"fold_{fold}_validation.csv",
            index=False
        )

        logging.info(
            "Fold %d | Training: %d | Validation: %d",
            fold, len(train_data), len(val_data)
        )


def main():
    logging.info("Loading dataset...")
    ratings = load_ratings()
    validate_ratings(ratings)

    logging.info("Creating 80/20 holdout split...")
    development, test = create_holdout_split(ratings)

    development.to_csv(
        SPLIT_DIR / "development.csv", index=False
    )

    test.to_csv(
        SPLIT_DIR / "final_test.csv", index=False
    )

    logging.info("Development set: %d ratings", len(development))
    logging.info("Final test set: %d ratings", len(test))

    logging.info("Creating 5-fold cross-validation...")
    create_cv_folds(development)

    print("\n" + "=" * 50)
    print("DATA SPLITTING COMPLETED")
    print("=" * 50)
    print(f"Development ratings : {len(development):,}")
    print(f"Final test ratings   : {len(test):,}")
    print(f"Cross-validation     : {N_SPLITS} folds")
    print(f"Random seed          : {RANDOM_STATE}")
    print(f"Saved in             : {SPLIT_DIR}")


if __name__ == "__main__":
    main()