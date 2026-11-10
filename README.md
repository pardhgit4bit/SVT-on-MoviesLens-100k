# MovieLens 100K Matrix Completion

## Overview
This project studies **matrix completion** for recommender systems using the MovieLens 100K dataset. The goal is to estimate missing user–movie ratings from the ratings that are observed.

## Theory

### Matrix Completion
User ratings can be represented as a partially observed matrix, where rows correspond to users and columns correspond to movies. Since users typically rate only a small fraction of movies, most entries are missing. Matrix completion estimates these missing entries by exploiting low-rank structure in the rating matrix.

### Singular Value Thresholding (SVT)
SVT is an iterative matrix-completion method. It alternates between:
1. Applying soft-thresholding to the singular values to encourage a low-rank estimate.
2. Updating the estimate using the difference between observed ratings and their current estimates.

The threshold controls how strongly small singular values are suppressed.

### Bias-Adjusted SVT
Ratings can also reflect consistent user and movie preferences. A user may generally give higher or lower ratings, while a movie may generally receive higher or lower ratings.

The bias model represents a rating as:

\[
\hat r_{ui} = \mu + b_u + b_i
\]

where \(\mu\) is the global mean, \(b_u\) is the user bias, and \(b_i\) is the movie bias.

Bias-adjusted SVT first subtracts this baseline from observed ratings, then applies SVT to the resulting residual matrix. The estimated low-rank residual is added back to the bias baseline to produce the final prediction.

## Models
- **Global Mean:** predicts the overall average rating.
- **User–Item Bias:** accounts for global, user, and movie rating tendencies.
- **Original SVT:** applies singular value thresholding directly to the observed rating matrix.
- **Bias-Adjusted SVT:** applies SVT to residuals after removing user and movie biases.

## Evaluation
The models are evaluated using held-out ratings and five-fold cross-validation. Prediction quality is assessed using **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)**.

## References
- Cai, J.-F., Candès, E. J., & Shen, Z. (2010). *A Singular Value Thresholding Algorithm for Matrix Completion*. SIAM Journal on Optimization.
- Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems*. Computer.
