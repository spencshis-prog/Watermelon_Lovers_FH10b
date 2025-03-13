import os
import pandas as pd
import streamlit as st

# Important: set page config before creating any Streamlit elements
st.set_page_config(page_title="Unified Model Error Comparison Dashboard", layout="wide")

MODEL_DIRS = {
    "LR": "testing_lr",
    "RF": "testing_rf",
    "ET": "testing_et",
    "CAT": "testing_cat",
    "LGBM": "testing_lgbm",
    "XGB": "testing_xgb",
    "NN": "testing_nn"
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
        elif model_label in ["RF", "ET", "XGB", "CAT"]:
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


def main():
    st.title("Unified Model Error Comparison Dashboard")
    print("here")

    # Let user choose "Holdout" or "KFold", defaulting to "Holdout"
    report_type = st.selectbox("Select Report Type", ["Holdout", "KFold"], index=0)

    base_dir = os.getcwd()
    df = load_all_reports(base_dir, report_type=report_type)

    if df.empty:
        st.error("No reports found for the selected type.")
        return

    st.subheader("Combined Report Data")
    st.write("Below is a single table containing the metrics from all model types "
             "(LinearRegression, RandomForestRegressor, ExtraTreesRegressor, XGBoost, "
             "CatBoost, NeuralNetwork).")

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


if __name__ == "__main__":
    main()
