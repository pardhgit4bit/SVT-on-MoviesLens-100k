# src/exploration.py

from pathlib import Path
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_ratings, validate_ratings


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def save_plot(fig, filename):
    """Save a figure in the project's results directory."""
    output_path = FIGURES_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved: %s", output_path)


def explore_ratings(df):
    """Generate exploratory statistics and visualizations."""

    n_ratings = len(df)
    n_users = df["user_id"].nunique()
    n_movies = df["item_id"].nunique()

    total_possible = n_users * n_movies
    density = n_ratings / total_possible
    sparsity = 1 - density

    print("\n" + "=" * 50)
    print("MOVIELENS 100K - EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    print(f"Number of ratings : {n_ratings:,}")
    print(f"Number of users   : {n_users:,}")
    print(f"Number of movies  : {n_movies:,}")
    print(f"Average rating    : {df['rating'].mean():.4f}")
    print(f"Rating std. dev.  : {df['rating'].std():.4f}")
    print(f"Matrix dimensions : {n_users} x {n_movies}")
    print(f"Matrix density    : {density:.4%}")
    print(f"Matrix sparsity   : {sparsity:.4%}")

    # Aggregated statistics
    user_counts = df.groupby("user_id").size().rename("rating_count")
    movie_stats = df.groupby("item_id")["rating"].agg(
        rating_count="count",
        average_rating="mean"
    )

    # 1. Rating distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="rating", color="steelblue", ax=ax)
    ax.set_title("Distribution of Movie Ratings")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Ratings")
    save_plot(fig, "01_rating_distribution.png")

    # 2. Ratings per user
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(user_counts, bins=30, color="darkorange", ax=ax)
    ax.set_title("Distribution of Ratings per User")
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("Number of Users")
    save_plot(fig, "02_ratings_per_user.png")

    # 3. Ratings per movie
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        movie_stats["rating_count"],
        bins=30,
        color="seagreen",
        ax=ax
    )
    ax.set_title("Distribution of Ratings per Movie")
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("Number of Movies")
    save_plot(fig, "03_ratings_per_movie.png")

    # 4. Most-rated movies
    most_rated = movie_stats.sort_values(
        "rating_count", ascending=False
    ).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    most_rated["rating_count"].sort_values().plot(
        kind="barh", ax=ax, color="slateblue"
    )
    ax.set_title("15 Most-Rated Movies")
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("Movie ID")
    save_plot(fig, "04_most_rated_movies.png")

    # 5. Highest average ratings, considering movies with at least 50 ratings
    popular_movies = movie_stats[movie_stats["rating_count"] >= 50]
    highest_rated = popular_movies.sort_values(
        "average_rating", ascending=False
    ).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    highest_rated["average_rating"].sort_values().plot(
        kind="barh", ax=ax, color="mediumpurple"
    )
    ax.set_title("15 Highest-Rated Movies (At Least 50 Ratings)")
    ax.set_xlabel("Average Rating")
    ax.set_ylabel("Movie ID")
    save_plot(fig, "05_highest_rated_movies.png")

    # 6. Rating activity over time
    dates = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    monthly_counts = dates.dt.strftime("%Y-%m").value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(11, 5))
    monthly_counts.plot(kind="line", marker="o", ax=ax, color="teal")
    ax.set_title("Number of Ratings Over Time")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Number of Ratings")
    ax.tick_params(axis="x", rotation=45)
    save_plot(fig, "06_rating_activity_over_time.png")

    # 7. Matrix sparsity
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(["User-Movie Matrix"], [density * 100], label="Observed")
    ax.barh(
        ["User-Movie Matrix"],
        [sparsity * 100],
        left=[density * 100],
        label="Missing"
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of Matrix Entries")
    ax.set_title(f"Matrix Sparsity: {sparsity:.2%}")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.65), ncol=2)
    save_plot(fig, "07_matrix_sparsity.png")

    # Save aggregated statistics for later reference
    movie_stats.to_csv(TABLES_DIR / "movie_statistics.csv")
    user_counts.to_csv(TABLES_DIR / "user_activity.csv", header=True)

    print("\nEDA completed successfully.")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Tables saved to : {TABLES_DIR}")


def main():
    logging.info("Loading MovieLens 100K dataset...")
    ratings = load_ratings()

    logging.info("Validating dataset...")
    validate_ratings(ratings)

    explore_ratings(ratings)


if __name__ == "__main__":
    main()