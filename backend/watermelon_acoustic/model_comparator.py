import os
import pandas as pd
import streamlit as st
import numpy as np

# Important: set page config before creating any Streamlit elements
st.set_page_config(page_title="Unified Comparison Dashboard", layout="wide")

MODEL_DIRS = {
    "LR": "testing_lr",
    "RF": "testing_rf",
    "ET": "testing_et",
    "CAT": "testing_cat",
    "LGBM": "testing_lgbm",
    "XGB": "testing_xgb",
    "MLP": "testing_mlp",
    "KRS": "testing_krs"
}


def parse_report(report_file, model_label):
    """
    Parses a single report file (with blocks separated by blank lines) into a DataFrame.
    For LR, we expect the descriptor labeled 'Regularization:'.
    For RF/XGB, we expect the descriptor labeled 'Tuning:' or 'Hypertuning:'.
    For other models (e.g., NN), we also treat it as Tuning by default.
    """
    with open(report_file, "r") as f:
        content = f.read().strip()

    blocks = [blk.strip() for blk in content.split("\n\n") if blk.strip()]
    rows = []
    for block in blocks:
        lines = block.splitlines()
        # We expect at least 5 lines in each block: 1 descriptor line + 4 metric lines
        if len(lines) < 5:
            continue

        # Default placeholders
        nr = ""
        fe = ""
        regularization_val = "-"
        tuning_val = "-"

        # ---- PARSE THE DESCRIPTORS ----
        descriptors = [d.strip() for d in lines[0].split(",")]
        for desc in descriptors:
            if desc.startswith("Noise Reduction:"):
                nr = desc.split(":", 1)[1].strip()
            elif desc.startswith("Feature Extraction:"):
                fe = desc.split(":", 1)[1].strip()
            elif desc.startswith("Regularization:"):
                regularization_val = desc.split(":", 1)[1].strip()
            elif desc.startswith("Hyper-tuning:"):
                tuning_val = desc.split(":", 1)[1].strip()

        # ---- ASSIGN BASED ON MODEL TYPE ----
        if model_label == "LR":
            # LR uses regularization; no tuning
            reg = regularization_val
            tuning = "-"
        elif model_label in ["RF", "ET", "XGB", "CAT", "LGBM", "MLP", "KRS"]:
            # RF / XGB uses tuning; no regularization
            reg = "-"
            tuning = tuning_val
        else:
            # NN or other: assume it uses 'tuning' style
            reg = "-"
            tuning = tuning_val

        # ---- PARSE THE METRICS (lines[1..4]) ----
        mae = float(lines[1].split(":", 1)[1].strip())
        mse = float(lines[2].split(":", 1)[1].strip())
        rmse = float(lines[3].split(":", 1)[1].strip())
        r2 = float(lines[4].split(":", 1)[1].strip())

        rows.append({
            "Model": model_label,
            "Noise Reduction": nr,
            "Feature Extraction": fe,
            "Regularization": reg,
            "Tuning": tuning,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        })

    return pd.DataFrame(rows)


def load_all_reports(base_dir, report_type="Holdout"):
    """
    Loops over each model directory (LR, RF, XGB, NN), checks if the chosen report file
    (report_holdout.txt or report_kfold.txt) exists, parses it, and merges into one DataFrame.
    """
    report_filename = "report_holdout.txt" if report_type == "Holdout" else "report_kfold.txt"

    all_dfs = []
    progress_bar = st.progress(0)
    total_models = len(MODEL_DIRS)

    for idx, (model_label, subdir) in enumerate(MODEL_DIRS.items(), start=1):
        report_path = os.path.join(base_dir, "output", subdir, report_filename)
        if os.path.exists(report_path):
            print(f"[INFO] {report_filename} found for model: {model_label} at {report_path}")
            df_model = parse_report(report_path, model_label=model_label)
            all_dfs.append(df_model)
        else:
            print(f"[INFO] No {report_filename} found for model: {model_label} at {report_path}")

        progress = int(idx / total_models * 100)
        progress_bar.progress(progress)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=["Model", "Noise Reduction", "Feature Extraction",
                                     "Regularization", "Tuning", "MAE", "MSE", "RMSE", "R2"])


def run_model_comparison_table():
    st.title("Unified Model Error Comparison Dashboard")

    # Let user choose "Holdout" or "KFold", defaulting to "Holdout"
    report_type = st.selectbox("Select Report Type", ["Holdout", "KFold"], index=0)

    base_dir = os.getcwd()
    df = load_all_reports(base_dir, report_type=report_type)

    if df.empty:
        st.error("No reports found for the selected type.")
        return

    st.subheader("Combined Report Data")

    # Use container width to avoid horizontal scrolling
    st.dataframe(df, use_container_width=True)

    numeric_cols = ["MAE", "MSE", "RMSE", "R2"]
    sort_column = st.selectbox("Sort By", ["Default"] + numeric_cols)
    sort_order = st.selectbox("Sort Order", ["Ascending", "Descending", "Default"])

    if sort_column != "Default" and sort_order != "Default":
        ascending = (sort_order == "Ascending")
        df_sorted = df.sort_values(by=sort_column, ascending=ascending)
        st.subheader("Sorted Report Data")
        st.dataframe(df_sorted, use_container_width=True)
    else:
        st.write("Select a metric and order to sort the table, or choose 'Default' to keep the original order.")


def load_history(base_dir):
    """
    Reads the report_history.txt from each model folder and concatenates them into one DataFrame.
    Assumes that each file is CSV formatted with columns:
    Run,Model,Noise Reduction,Feature Extraction,Hyper-tuning,MAE,MSE,RMSE,R2
    """
    history_dfs = []
    for model_label, subdir in MODEL_DIRS.items():
        history_path = os.path.join(base_dir, "output", subdir, "report_history.txt")
        if os.path.exists(history_path):
            try:
                df_model = pd.read_csv(history_path)
                # Ensure the Model column exists; if not, add it.
                if "Model" not in df_model.columns:
                    df_model["Model"] = model_label
                history_dfs.append(df_model)
            except Exception as e:
                st.error(f"Error reading {history_path}: {e}")
    if history_dfs:
        return pd.concat(history_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def compare_runs(df_history, run_n, run_m):
    """
    Given a history DataFrame and two run numbers (run_n and run_m),
    computes for each unique combination (Model, Noise Reduction, Feature Extraction, Hyper-tuning)
    the differences and squared differences for each metric.

    If a value is missing for run n or run m for a combination, the result will be NaN.
    Returns a DataFrame with the comparison results.
    """
    # Pivot the history so that each row is uniquely identified by the key columns and columns are metrics for each run.
    key_cols = ["Model", "Noise Reduction", "Feature Extraction", "Hyper-tuning"]
    metrics = ["MAE", "MSE", "RMSE", "R2"]

    # Create a MultiIndex pivot table: rows are the key columns, columns are (metric, Run)
    df_pivot = df_history.pivot_table(index=key_cols, columns="Run", values=metrics)
    df_pivot = df_pivot.reset_index()

    # For convenience, flatten the column names so that metric for a specific run is like MAE_1, MSE_1, etc.
    df_pivot.columns = [f"{col[0]}_{col[1]}" if col[1] != "" else col[0] for col in df_pivot.columns.values]

    # For each metric, compute the difference: diff = value in run_m minus value in run_n,
    # and its squared difference.
    for metric in metrics:
        col_n = f"{metric}_{run_n}"
        col_m = f"{metric}_{run_m}"
        diff_col = f"{metric}_diff"
        diff_sq_col = f"{metric}_diff_sq"

        # Use .get() to safely handle missing columns
        def safe_diff(row):
            val_n = row.get(col_n, np.nan)
            val_m = row.get(col_m, np.nan)
            try:
                return val_m - val_n
            except Exception:
                return np.nan

        df_pivot[diff_col] = df_pivot.apply(safe_diff, axis=1)
        df_pivot[diff_sq_col] = df_pivot[diff_col] ** 2

    return df_pivot


def run_version_comparison_table(base_dir):
    st.subheader("Version Comparison Dashboard")
    df_history = load_history(base_dir)
    if df_history.empty:
        st.warning("No historical run data available.")
        return

    # Get unique run numbers from the history; these should be numeric.
    runs = sorted(df_history["Run"].unique(), reverse=True)
    if not runs:
        st.warning("No run numbers found in history.")
        return

    # Create dropdowns for run n and run m.
    run_n = st.selectbox("Select Run N", runs, index=0)  # highest first (most recent) by default
    run_m_option = st.selectbox("Select Run M", ["Most Recent"] + runs, index=0)
    # If "Most Recent" is chosen, set run_m to the maximum run available.
    if run_m_option == "Most Recent":
        run_m = max(runs)
    else:
        run_m = run_m_option

    st.write(f"Comparing results from Run {run_n} with Run {run_m}.")

    # Create the comparative DataFrame.
    df_comparison = compare_runs(df_history, run_n, run_m)

    # For each key combination (each row), we might want to indicate if a model was created or deleted.
    # For example, if the value for run_n is NaN but exists in run_m, label it as "Created (n/a)".
    # Similarly, if it exists in run_n but not in run_m, label as "Deleted (n/a)".
    # We'll add additional columns for one metric (e.g. MAE) as an example.
    def status(row, metric):
        val_n = row.get(f"{metric}_{run_n}", np.nan)
        val_m = row.get(f"{metric}_{run_m}", np.nan)
        if pd.isna(val_n) and not pd.isna(val_m):
            return "Created (n/a)"
        elif not pd.isna(val_n) and pd.isna(val_m):
            return "Deleted (n/a)"
        else:
            return ""

    df_comparison["MAE_status"] = df_comparison.apply(lambda row: status(row, "MAE"), axis=1)

    st.dataframe(df_comparison)


if __name__ == "__main__":
    st.subheader("Regressor Classes")
    st.write("Comparison of ensemble tree models (RandomForest, ExtraTrees), boosting models (XGBoost, "
             "LightGBM, CatBoost), and neural networks (MultiLayerPerceptron, Keras)")

    st.subheader("Hyperparameter Tuning")
    st.write("Comparison of default, grid search, random search, bayesian optimization, and optuna optimization")

    base_dir = os.getcwd()
    run_model_comparison_table()
    run_version_comparison_table(base_dir)
