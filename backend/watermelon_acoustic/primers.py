# primers.py
import numpy as np
from sklearn.preprocessing import QuantileTransformer


def remove_outliers(X, y, fnames, threshold=3.0):
    """
    Removes outliers by clipping feature values that fall outside a specified number
    of standard deviations (default 3). This can be beneficial for tree-based models
    like Random Forest and Extra Trees if extreme values exist.

    Parameters:
      X: 2D numpy array of features.
      y: Target values.
      fnames: List of filenames.
      threshold: Number of standard deviations for clipping (default 3.0).

    Returns:
      (X_clipped, y, fnames)
    """
    X = np.array(X, dtype=np.float64)
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    lower_bound = means - threshold * stds
    upper_bound = means + threshold * stds
    X_clipped = np.clip(X, lower_bound, upper_bound)
    return X_clipped, y, fnames


def log_transform(X, y, fnames):
    """
    Applies a logarithmic transformation to features that are strictly positive.
    Often beneficial for boosting models (XGBoost, LightGBM, CatBoost) when feature
    distributions are heavy-tailed.

    Parameters:
      X: 2D numpy array of features.
      y: Target values.
      fnames: List of filenames.

    Returns:
      (X_log, y, fnames)
    """
    X = np.array(X, dtype=np.float64)
    epsilon = 1e-6  # to avoid log(0)
    X_trans = X.copy()
    for i in range(X.shape[1]):
        if np.all(X[:, i] > 0):
            X_trans[:, i] = np.log(X[:, i] + epsilon)
    return X_trans, y, fnames


def standard_scale(X, y, fnames):
    """
    Standardizes features by removing the mean and scaling to unit variance.
    This is helpful for boosting models (XGBoost, LightGBM) when features are on
    very different scales.

    Parameters:
      X: 2D numpy array.
      y: Target values.
      fnames: List of filenames.

    Returns:
      (X_scaled, y, fnames)
    """
    X = np.array(X, dtype=np.float64)
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1.0  # prevent division by zero
    X_scaled = (X - means) / stds
    return X_scaled, y, fnames


def quantile_transform(X, y, fnames, n_quantiles=1000, output_distribution='uniform'):
    """
    Applies a quantile transformation to features. This rank-based transformation
    maps the data to a uniform (or normal) distribution and is useful for tree-based
    models (Random Forest, Extra Trees) when feature distributions are highly skewed.

    Parameters:
      X: 2D numpy array.
      y: Target values.
      fnames: List of filenames.
      n_quantiles: Number of quantiles to use (default 1000).
      output_distribution: "uniform" or "normal" (default "uniform").

    Returns:
      (X_transformed, y, fnames)
    """
    X = np.array(X, dtype=np.float64)
    transformer = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution, random_state=42)
    X_transformed = transformer.fit_transform(X)
    return X_transformed, y, fnames


# Define the primers dictionary.
# For Random Forest and Extra Trees:
#   1. remove_outliers: Clip extreme values to reduce influence of outliers.
#   2. quantile_transform: Map features to a uniform distribution.
#
# For XGBoost and LightGBM:
#   1. log_transform: Compress heavy-tailed distributions.
#   2. standard_scale: Normalize features to zero mean and unit variance.
#
# For CatBoost:
#   A simple log_transform might be sufficient.
primers = {
    "rf": [remove_outliers, quantile_transform],
    "et": [remove_outliers, quantile_transform],
    "xgb": [log_transform, standard_scale],
    "lgbm": [log_transform, standard_scale],
    "cat": [log_transform]
}
