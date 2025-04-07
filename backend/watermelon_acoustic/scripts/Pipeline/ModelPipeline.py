import os

import functions
from scripts.Pipeline.testing import test_all
from scripts.Pipeline.training import train_all


class ModelPipeline:
    def __init__(self, model_tag, model_cls, dataset_path, primer_functions=None, inner_folds=3, outer_folds=5, early_stopping_rounds=None, eval_set_split=None,
                 ht_options=None, params_grid=None, params_random=None, params_bayesian=None, params_optuna=None):
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
        if ht_options is None:
            ht_options = ["default"]
        self.model_tag = model_tag
        self.model_cls = model_cls
        self.inner_folds = inner_folds
        self.outer_folds = outer_folds
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_set_split = eval_set_split
        self.ht_options = ht_options  # list of strings
        self.params_grid = params_grid
        self.params_random = params_random
        self.params_bayesian = params_bayesian
        self.params_optuna = params_optuna

        if primer_functions is None:
            self.primer_functions = []
        else:
            self.primer_functions = primer_functions

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.fe_base_dir = dataset_path

        self.models_output_dir = os.path.join(self.base_dir, "../../output", f"models_{model_tag}")
        self.testing_output_dir = os.path.join(self.base_dir, "../../output", f"testing_{model_tag}")

        self.report_kfold_path = os.path.join(self.testing_output_dir, "report_kfold.txt")
        self.report_holdout_path = os.path.join(self.testing_output_dir, "report_holdout.txt")
        self.report_params_path = os.path.join(self.testing_output_dir, "report_params.txt")

    def prime(self, X, y, fnames, training=True):
        """
        Applies model-specific priming functions to the data.
        Wraps data in a dict if necessary. In training mode, returns:
          (X_transformed, y, fnames, transformers)
        Otherwise, returns:
          (X_transformed, y, fnames)
        """
        if not isinstance(X, dict):
            data = {"X": X, "y": y, "fnames": fnames, "primed": False, "transformers": {}}
        else:
            data = X

        if data.get("primed", False):
            if training:
                return data["X"], data["y"], data["fnames"], data["transformers"]
            else:
                return data["X"], data["y"], data["fnames"]

        primer_names = [p[0] if isinstance(p, tuple) else p.__name__ for p in self.primer_functions]
        print(
            f"[{self.model_tag.upper()}] Priming {os.path.relpath(self.fe_base_dir, os.getcwd())} using {len(self.primer_functions)} primer(s): {', '.join(primer_names)}")

        X_out, y_out, fnames_out = data["X"], data["y"], data["fnames"]
        transformers = {}

        for primer in self.primer_functions:
            # If the primer is not a tuple, get its name from __name__
            if isinstance(primer, tuple):
                primer_name, primer_fn = primer
            else:
                primer_name = primer.__name__
                primer_fn = primer

            # If in training and the primer is stateful (name ends with '_fit'), expect four outputs.
            if training and primer_name.endswith('_fit'):
                X_out, y_out, fnames_out, fitted_trans = primer_fn(X_out, y_out, fnames_out)
                transformers[primer_name] = fitted_trans
            else:
                X_out, y_out, fnames_out = primer_fn(X_out, y_out, fnames_out)

        data["X"], data["y"], data["fnames"] = X_out, y_out, fnames_out
        data["transformers"] = transformers
        data["primed"] = True

        if training:
            return X_out, y_out, fnames_out, transformers
        else:
            return X_out, y_out, fnames_out

    def train(self):
        # clear training outputs (kfold error metrics + models)
        open(self.report_kfold_path, "w").close()
        functions.clear_output_directory(self.models_output_dir)

        for ht in sorted(self.ht_options):
            functions.green_print(f"\n=== Training {self.model_tag} models with hyperparameter tuning: {ht} ===")
            if ht == "optuna":
                print("optuna")
            else:
                train_all(self, ht)

    def test(self, run_num):
        open(self.report_holdout_path, "w").close()
        open(self.report_params_path, "w").close()

        functions.generate_report_params(self.models_output_dir, self.report_params_path)

        functions.clear_output_directory(os.path.join(self.testing_output_dir, "heatmaps"))
        functions.clear_output_directory(os.path.join(self.testing_output_dir, "predicted_vs_actual"))
        functions.clear_output_directory(os.path.join(self.testing_output_dir, "residual_plots"))

        print(f"\n=== Testing {self.model_tag} models on hold-out test sets ===")
        test_all(self, run_num)
        print(f"{self.model_tag.upper()} pipeline completed.")

    def get_params_dict(self, ht):
        """
        Returns the appropriate parameter dictionary for the given hyper-tuning method.
        """
        if ht == "grid":
            return self.params_grid
        elif ht == "random":
            return self.params_random
        elif ht == "bayesian":
            return self.params_bayesian
        elif ht == "optuna":
            return self.params_optuna
        else:
            return None
