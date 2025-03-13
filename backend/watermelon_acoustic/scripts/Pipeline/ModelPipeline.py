

class ModelPipeline:
    def __init__(self, model_name, model_cls, inner_kfolding=3, outer_kfolding=5, hyper_tuning=["default"],
                 params_grid=None, params_random=None, params_bayesian=None, params_optuna=None):
        """
        model_name: String identifier, e.g. "RF", "XGB", etc.
        model_cls: The regression model class (e.g. RandomForestRegressor, XGBRegressor)
        inner_kfolding: Number of folds to use for hyperparameter tuning (inner CV).
        outer_kfolding: Number of folds for outer cross-validation (final evaluation).
        hyper_tuning: List of tuning strategies to cycle through, e.g. ["default", "grid", "random", "optuna"]
        params_grid: Dictionary of parameters for grid search.
        params_random: Dictionary of parameters for random search.
        params_bayesian: Dictionary of parameters for Bayesian search.
        params_optuna: Dictionary (or search space preset) for Optuna tuning.
        """
        self.model_name = model_name
        self.model_cls = model_cls
        self.inner_kfolding = inner_kfolding
        self.outer_kfolding = outer_kfolding
        self.hyper_tuning = hyper_tuning  # list of strings
        self.params_grid = params_grid
        self.params_random = params_random
        self.params_bayesian = params_bayesian
        self.params_optuna = params_optuna