import os
import shutil

import joblib
import numpy as np
from collections import defaultdict

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import params


def clear_output_directory(output_dir):
    """
    Removes the output directory if it exists and creates a new empty one.
    """
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
        except PermissionError as e:
            print(f"PermissionError while removing {output_dir}: {e}")
            # Try to remove files inside the directory
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    try:
                        os.remove(os.path.join(root, file))
                    except Exception as e:
                        print(f"Could not remove file {file}: {e}")
    os.makedirs(output_dir, exist_ok=True)


def combine_folders(folder1, folder2, output_folder):
    """
    Combines all files from folder1 and folder2 into output_folder.
    """
    clear_output_directory(output_folder)
    for folder in [folder1, folder2]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                src = os.path.join(folder, file)
                dst = os.path.join(output_folder, file)
                shutil.copy(src, dst)
    print(f"Combined folders {folder1} and {folder2} into {output_folder}")


def green_print(message):
    print("\033[92m" + message + "\033[0m")


def load_feature_data(folder):
    """
    Loads .npy feature files from folder.
    Assumes naming format: <watermelonID>_<brix>_<index>.npy.
    Returns X, y, fnames.
    """
    data, labels, fnames = [], [], []
    for file in os.listdir(folder):
        if file.lower().endswith(".npy"):
            parts = file.split("_")
            if len(parts) < 3:
                continue
            try:
                brix_val = float(parts[1])
            except:
                continue
            file_path = os.path.join(folder, file)
            feat = np.load(file_path)
            data.append(feat)
            labels.append(brix_val)
            fnames.append(file)
    return np.array(data), np.array(labels), fnames

def load_feature_data_wrapped(folder):
    """
    Wraps the loaded feature data in a dictionary with a 'primed' flag.
    """
    X, y, fnames = load_feature_data(folder)
    return {"X": X, "y": y, "fnames": fnames, "primed": False}


def relevant_params(model, model_type, hyper_tuning):
    """
    Returns a filtered dictionary of the model's parameters based on model_type.
    If hyper_tuning is not 'default' and the model has a best_params_ attribute,
    that dictionary is used; otherwise, model.get_params() is used.
    """
    model_type = model_type.lower()

    # Get the parameter dictionary from best_params_ if available.
    if hyper_tuning != "default" and hasattr(model, "best_params_"):
        params_dict = model.best_params_
    elif hasattr(model, "get_params"):
        params_dict = model.get_params()
    else:
        params_dict = {}

    if model_type in ["rf", "et"]:
        keys = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']
    elif model_type == "xgb":
        keys = ['n_estimators', 'max_depth', 'learning_rate', 'gamma', 'reg_alpha', 'reg_lambda', 'subsample']
    elif model_type == "cat":
        keys = ['iterations', 'depth', 'learning_rate', 'l2_leaf_reg']
    else:
        keys = list(params_dict.keys())

    filtered = {k: params_dict[k] for k in keys if k in params_dict}
    return filtered


def load_model(model_path):
    """
    Loads a saved model (.pkl) and ensures compatibility with GridSearchCV,
    RandomizedSearchCV, and BayesSearchCV.

    Args:
        model_path (str): Path to the .pkl model file.

    Returns:
        model: The loaded model (best estimator if applicable).
    """
    try:
        model = joblib.load(model_path)

        # Handle GridSearchCV, RandomizedSearchCV, or BayesSearchCV
        if isinstance(model, (GridSearchCV, RandomizedSearchCV, BayesSearchCV)):
            print(f"[INFO] Found tuned model: {model.__class__.__name__}")
            return model.best_estimator_ if hasattr(model, 'best_estimator_') else model

        # Handle standard models
        if isinstance(model, (RandomForestRegressor, ExtraTreesRegressor, LGBMRegressor,
                              XGBRegressor, CatBoostRegressor)):
            return model

        print(f"[WARNING] Unknown model type for file: {model_path}")
        return model

    except Exception as e:
        print(f"[ERROR] Failed to load model from {model_path}: {e}")
        return None


def generate_report_params(model_output_dir, report_path):
    """
    Generates a `report_params.txt` file summarizing:
    - Parameter ranges set for tuning
    - Chosen parameter values across models
    - Model summary listing each model's best parameters
    """
    param_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    model_summary = {}

    # Iterate through each model file
    for model_file in os.listdir(model_output_dir):
        if model_file.endswith('.pkl'):
            model_path = os.path.join(model_output_dir, model_file)
            model_name = model_file.split('_')[0]  # RF / ET / XGB / CAT
            hyper_tuning = model_file.split('_')[-1].replace('.pkl', '')  # e.g. grid / random / bayesian

            # Load the saved model
            model = load_model(model_path)
            if model is None:
                continue  # Skip failed loads

            # Extract model parameters
            if hasattr(model, 'best_params_'):
                model_params = model.best_params_
            else:
                model_params = model.get_params()

            # Identify correct param range; if bayesian, fallback to using the search space dict.
            if hyper_tuning == 'bayesian':
                param_dict_key = f"{model_name.lower()}_search_spaces"
            else:
                param_dict_key = f"{model_name.lower()}_{hyper_tuning}"
            param_ranges = getattr(params, param_dict_key, {})

            # Initialize model summary
            model_summary[model_file] = {}

            # Track chosen values
            for param, possible_values in param_ranges.items():
                chosen_value = model_params.get(param, None)
                model_summary[model_file][param] = chosen_value

                # Handle both discrete (list) and continuous (tuple) parameter types
                if isinstance(possible_values, list):
                    # Discrete values
                    for value in possible_values:
                        if value == chosen_value:
                            param_counts[hyper_tuning][param][value] += 1
                        else:
                            param_counts[hyper_tuning][param][value] += 0
                elif isinstance(possible_values, tuple):
                    # Continuous values — use range notation
                    lower, upper = possible_values[:2]
                    key_str = f"{chosen_value}"
                    param_counts[hyper_tuning][param][key_str] += 1

    # Write the report
    with open(report_path, 'w') as f:
        f.write(f"=== {model_name} Parameter Tuning Report ===\n")

        for tuning_type, params_dict in param_counts.items():
            f.write(f"\n--- {tuning_type.upper()} ---\n")
            for param, value_counts in params_dict.items():
                total = sum(value_counts.values())
                # Determine the correct key for parameter ranges.
                if tuning_type == "bayesian":
                    key = f"{model_name.lower()}_search_spaces"
                else:
                    key = f"{model_name.lower()}_{tuning_type}"
                param_range = getattr(params, key, {}).get(param, None)

                # For continuous ranges, show the range directly; otherwise, show the list.
                if isinstance(param_range, tuple):
                    range_str = f"({param_range[0]} to {param_range[1]})"
                else:
                    range_str = param_range if param_range else "N/A"
                f.write(f"\n{model_name.upper()} | {tuning_type} | {param} | {range_str}\n")
                for value, count in value_counts.items():
                    f.write(f"    {value}: {count}/{total}\n")

        # Append Model Summary
        f.write("\n=== Model Summary ===\n")
        for m_name, chosen_params in model_summary.items():
            f.write(f"{m_name} -> {chosen_params}\n")

    print(f"Report generated at: {report_path}")
