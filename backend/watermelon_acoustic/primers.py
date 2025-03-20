import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler


# --- Stateful Transformer for Outlier Removal ---
class RemoveOutliersTransformer:
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.means_ = None
        self.stds_ = None

    def fit(self, X, y=None):
        X = np.array(X, dtype=np.float64)
        self.means_ = np.mean(X, axis=0)
        self.stds_ = np.std(X, axis=0)
        return self

    def transform(self, X, y=None):
        X = np.array(X, dtype=np.float64)
        lower_bound = self.means_ - self.threshold * self.stds_
        upper_bound = self.means_ + self.threshold * self.stds_
        X_clipped = np.clip(X, lower_bound, upper_bound)
        return X_clipped

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# --- Stateless Log Transformation ---
def log_transform(X, y, fnames):
    """
    Log transform is stateless and applied directly.
    """
    X = np.array(X, dtype=np.float64)
    epsilon = 1e-6  # to avoid log(0)
    X_trans = X.copy()
    for i in range(X.shape[1]):
        # Only apply if the entire feature is positive
        if np.all(X[:, i] > 0):
            X_trans[:, i] = np.log(X[:, i] + epsilon)
    return X_trans, y, fnames


# --- Stateful Transformer for Standard Scaling ---
class StandardScaleTransformer:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X, y=None):
        return self.scaler.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# --- Stateful Transformer for Quantile Transformation ---
class QuantileTransformWrapper:
    def __init__(self, n_quantiles=1000, output_distribution='uniform'):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.transformer = QuantileTransformer(n_quantiles=self.n_quantiles,
                                               output_distribution=self.output_distribution,
                                               random_state=42)

    def fit(self, X, y=None):
        self.transformer.fit(X)
        return self

    def transform(self, X, y=None):
        return self.transformer.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# --- Wrappers to provide a common interface to ModelPipeline ---
# Each primer now returns the transformed data as well as (optionally) the fitted transformer.
def remove_outliers_fit(X, y, fnames, threshold=3.0):
    transformer = RemoveOutliersTransformer(threshold=threshold)
    X_trans = transformer.fit_transform(X)
    return X_trans, y, fnames, transformer


def quantile_transform_fit(X, y, fnames, n_quantiles=1000, output_distribution='uniform'):
    transformer = QuantileTransformWrapper(n_quantiles=n_quantiles,
                                           output_distribution=output_distribution)
    X_trans = transformer.fit_transform(X)
    return X_trans, y, fnames, transformer


def standard_scale_fit(X, y, fnames):
    transformer = StandardScaleTransformer()
    X_trans = transformer.fit_transform(X)
    return X_trans, y, fnames, transformer


# You might similarly write “transform” versions for when you have already fitted transformers:
def remove_outliers(X, transformer):
    return transformer.transform(X)


def quantile_transform(X, transformer):
    return transformer.transform(X)


def standard_scale(X, transformer):
    return transformer.transform(X)


# --- Primer Dictionary ---
# The keys remain the same but each element is now a tuple.
# For each model type, you can adjust which primers to apply.
primers = {
    "rf": [("remove_outliers", remove_outliers_fit),
           ("quantile_transform", quantile_transform_fit)],
    "et": [("remove_outliers", remove_outliers_fit),
           ("quantile_transform", quantile_transform_fit)],
    "xgb": [("log_transform", log_transform),
            ("standard_scale", standard_scale_fit)],
    "lgbm": [("log_transform", log_transform),
             ("standard_scale", standard_scale_fit)],
    "cat": [("log_transform", log_transform)]
}
