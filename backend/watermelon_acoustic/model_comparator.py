import os
import pandas as pd
import streamlit as st

MODEL_DIRS = {
    "LR": "testing_lr",
    "RF": "testing_rf",
    "XGB": "testing_xgb",
    "NN": "testing_nn"
}
# Each of these directories may contain "report_holdout.txt" and/or "report_kfold.txt".

def parse_report(report_file, model_label):
    """
    Parses a single report file (with blocks separated by blank lines) into a DataFrame.
    The data in the file is in blocks like:

      Noise Reduction: bandpass, Feature Extraction: mfcc, Regularization: ElasticNet
      MAE: 2.1412
      MSE: 6.3278
      RMSE: 2.5155
      R2: -5.7915

    We return a DataFrame with columns:
      [Model, Noise Reduction, Feature Extraction, Regularization/Tuning, MAE, MSE, RMSE, R2]
    'model_label' is something like "LR", "RF", "XGB", or "NN".
    """
    with open(report_file, "r") as f:
        content = f.read().strip()

    blocks = [blk.strip() for blk in content.split("\n\n") if blk.strip()]
    rows = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 5:
            continue
        # First line: parse descriptors
        # e.g. "Noise Reduction: bandpass, Feature Extraction: mfcc, Regularization: ElasticNet"
        descriptors = lines[0].split(",")
        nr = descriptors[0].split(":")[1].strip() if "Noise Reduction:" in descriptors[0] else ""
        fe = descriptors[1].split(":")[1].strip() if len(descriptors) > 1 and "Feature Extraction:" in descriptors[1] else ""
        reg = descriptors[2].split(":")[1].strip() if len(descriptors) > 2 else ""

        mae = float(lines[1].split(":")[1].strip())
        mse = float(lines[2].split(":")[1].strip())
        rmse = float(lines[3].split(":")[1].strip())
        r2 = float(lines[4].split(":")[1].strip())

        rows.append({
            "Model": model_label,
            "Noise Reduction": nr,
            "Feature Extraction": fe,
            "Regularization/Tuning": reg,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        })
    df = pd.DataFrame(rows)
    return df

def load_all_reports(base_dir, report_type="Holdout"):
    """
    Loops over each model directory (LR, RF, XGB, NN), checks if the chosen report file
    (report_holdout.txt or report_kfold.txt) exists, parses it, and merges into one DataFrame.
    """
    # "report_holdout.txt" or "report_kfold.txt"
    report_filename = "report_holdout.txt" if report_type == "Holdout" else "report_kfold.txt"

    all_dfs = []
    for model_label, subdir in MODEL_DIRS.items():
        report_path = os.path.join(base_dir, "output", subdir, report_filename)
        if os.path.exists(report_path):
            df_model = parse_report(report_path, model_label=model_label)
            all_dfs.append(df_model)
        else:
            # If a file doesn't exist, we skip it
            print(f"[INFO] No {report_filename} found for model: {model_label} at {report_path}")

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=["Model","Noise Reduction","Feature Extraction",
                                     "Regularization/Tuning","MAE","MSE","RMSE","R2"])

def main():
    st.title("Unified Model Error Comparison Dashboard")

    # Let user choose "Holdout" or "KFold", defaulting to "Holdout"
    report_type = st.selectbox("Select Report Type", ["Holdout", "KFold"], index=0)

    base_dir = os.getcwd()
    df = load_all_reports(base_dir, report_type=report_type)

    if df.empty:
        st.error("No reports found for the selected type.")
        return

    # Single table with a "Model" column
    st.subheader("Combined Report Data")
    st.write("Below is a single table containing the metrics from all model types (LR, RF, XGB, NN).")
    st.dataframe(df)

    # Sorting controls
    numeric_cols = ["MAE", "MSE", "RMSE", "R2"]
    sort_column = st.selectbox("Sort By", ["Default"] + numeric_cols)
    sort_order = st.selectbox("Sort Order", ["Ascending", "Descending", "Default"])

    # We create only one table
    if sort_column != "Default" and sort_order != "Default":
        ascending = (sort_order == "Ascending")
        df_sorted = df.sort_values(by=sort_column, ascending=ascending)
        st.subheader("Sorted Report Data")
        st.dataframe(df_sorted)
    else:
        st.write("Select a metric and order to sort the table, or choose 'Default' to keep the original order.")

if __name__ == "__main__":
    main()
