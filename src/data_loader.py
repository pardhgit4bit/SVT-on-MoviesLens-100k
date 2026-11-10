
"""
MovieLens 100K - Dataset Loading and Validation

Loads the original MovieLens 100K ratings file and validates
its structure before any preprocessing or experimentation.
"""

from pathlib import Path
import logging

import pandas as pd


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RATINGS_FILE = DATA_DIR / "u.data"

EXPECTED_ROWS = 100_000
EXPECTED_USERS = 943
EXPECTED_MOVIES = 1682

RATING_COLUMNS = [
    "user_id",
    "item_id",
    "rating",
    "timestamp",
]


# --------------------------------------------------
# 2. Logging configuration
# --------------------------------------------------

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 3. Load ratings
# --------------------------------------------------

def load_ratings(file_path: Path = RATINGS_FILE) -> pd.DataFrame:
    """
    Load MovieLens 100K ratings.

    File format:
        user_id    item_id    rating    timestamp

    Returns:
        A pandas DataFrame containing the ratings.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Ratings file not found: {file_path}"
        )

    logger.info("Loading ratings from: %s", file_path)

    ratings = pd.read_csv(
        file_path,
        sep="\t",
        names=RATING_COLUMNS,
        header=None,
        dtype={
            "user_id": "int32",
            "item_id": "int32",
            "rating": "int8",
            "timestamp": "int64",
        },
    )

    logger.info(
        "Successfully loaded %s ratings.",
        len(ratings),
    )

    return ratings


# --------------------------------------------------
# 4. Validate dataset
# --------------------------------------------------

def validate_ratings(
    ratings: pd.DataFrame,
    strict: bool = True,
) -> bool:
    """
    Validate the MovieLens ratings DataFrame.

    Strict mode additionally checks the expected
    dimensions of the official MovieLens 100K dataset.
    """

    logger.info("Starting dataset validation...")

    # Check columns
    if list(ratings.columns) != RATING_COLUMNS:
        raise ValueError(
            f"Unexpected columns: {list(ratings.columns)}"
        )

    # Check missing values
    if ratings.isnull().any().any():
        raise ValueError(
            "Dataset contains missing values."
        )

    # Check empty dataset
    if ratings.empty:
        raise ValueError("Dataset is empty.")

    # Check user IDs
    if not ratings["user_id"].between(1, EXPECTED_USERS).all():
        raise ValueError("Invalid user IDs detected.")

    # Check movie IDs
    if not ratings["item_id"].between(1, EXPECTED_MOVIES).all():
        raise ValueError("Invalid movie IDs detected.")

    # Check rating range
    if not ratings["rating"].between(1, 5).all():
        raise ValueError("Ratings must be between 1 and 5.")

    # Check integer ratings
    if not (ratings["rating"] % 1 == 0).all():
        raise ValueError("Ratings must be integers.")

    # Check timestamps
    if (ratings["timestamp"] < 0).any():
        raise ValueError("Invalid timestamps detected.")

    # Check duplicate user-movie pairs
    duplicates = ratings.duplicated(
        subset=["user_id", "item_id"]
    ).sum()

    if duplicates > 0:
        raise ValueError(
            f"Found {duplicates} duplicate user-movie pairs."
        )

    # Check canonical MovieLens 100K dimensions
    if strict:
        if len(ratings) != EXPECTED_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_ROWS} ratings, "
                f"found {len(ratings)}."
            )

        if ratings["user_id"].nunique() != EXPECTED_USERS:
            raise ValueError("Unexpected number of users.")

        if ratings["item_id"].nunique() != EXPECTED_MOVIES:
            raise ValueError("Unexpected number of movies.")

    logger.info("Dataset validation successful!")

    return True


# --------------------------------------------------
# 5. Dataset summary
# --------------------------------------------------

def display_dataset_summary(ratings: pd.DataFrame) -> None:
    """Display basic dataset statistics."""

    print("\n" + "=" * 55)
    print("MOVIELENS 100K - DATASET SUMMARY")
    print("=" * 55)

    print(f"Total ratings : {len(ratings):,}")
    print(f"Unique users  : {ratings['user_id'].nunique():,}")
    print(f"Unique movies : {ratings['item_id'].nunique():,}")

    print(
        f"Rating range  : "
        f"{ratings['rating'].min()} - "
        f"{ratings['rating'].max()}"
    )

    print(
        f"Average rating: "
        f"{ratings['rating'].mean():.4f}"
    )

    print(
        f"Rating std    : "
        f"{ratings['rating'].std():.4f}"
    )

    print("\nRating distribution:")
    print(
        ratings["rating"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nFirst five records:")
    print(ratings.head().to_string(index=False))

    print("\n" + "=" * 55)


# --------------------------------------------------
# 6. Main execution
# --------------------------------------------------

def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    ratings = load_ratings()

    validate_ratings(ratings, strict=True)

    display_dataset_summary(ratings)


if __name__ == "__main__":
    main()
