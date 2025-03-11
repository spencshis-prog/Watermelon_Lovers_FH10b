import os
import shutil
import numpy as np


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


def relevant_params(params, model_type, hyper_tuning):
    """
    Filters the full parameters dictionary for a model and hyper_tuning setting,
    returning only the most relevant parameters that you want to report.

    Parameters:
      params (dict): The full parameters dictionary from final_model.get_params()
                     or final_model.best_params_.
      model_type (str): The type of model. Supported values are:
                        "rf" for RandomForest,
                        "et" for ExtraTrees,
                        "xgb" for XGBoost,
                        "cat" for CatBoost.
      hyper_tuning (str): The hyperparameter tuning strategy used (e.g., "default", "grid",
                          "random", or "bayesian"). This can be used to further adjust which
                          keys to show if desired.

    Returns:
      dict: A filtered dictionary with only the relevant parameter keys.
    """
    model_type = model_type.lower()

    if model_type == "rf" or model_type == "et":
        # For RandomForest and ExtraTrees, we consider these parameters:
        keys = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']
    elif model_type == "xgb":
        # For XGBoost, these are the parameters we're primarily interested in.
        keys = ['n_estimators', 'max_depth', 'learning_rate', 'gamma', 'reg_alpha', 'reg_lambda', 'subsample']
    elif model_type == "cat":
        # For CatBoost, we consider:
        keys = ['iterations', 'depth', 'learning_rate', 'l2_leaf_reg']
    else:
        # Fallback: return all parameters.
        keys = list(params.keys())

    # Optionally, you can adjust keys further based on the hyper_tuning option.
    # For instance, if hyper_tuning == "default", you might want to only show a subset.
    # For now, we'll just filter based on the keys list.
    filtered = {k: params[k] for k in keys if k in params}
    return filtered
