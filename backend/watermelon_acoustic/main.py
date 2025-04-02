# TODO: pip install --upgrade "scikit-learn==1.2.2" "scikit-optimize==0.9.0" "scikeras==0.10.0"
# TODO: pip install numpy==1.19.5
# the above must be run antecedent to bayesian optimization on kerasregressor
# TODO: streamlit run main.py

import os

from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from skopt.learning import RandomForestRegressor
from xgboost import XGBRegressor

from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

import functions
import model_comparator
import params
import primers
from MLPWrapper import MLPWrapper

from scripts.Pipeline.ModelPipeline import ModelPipeline
from scripts.linear_regression import pipeline_lr
from scripts.preprocessing import pipeline_pp

import lightgbm as lgb

import warnings

warnings.filterwarnings("ignore", message="n_quantiles.*")

# Dataset Configuration
USE_QILIN = True
USE_LAB = False
TEST_SPLIT_RATIO = 0.15  # fraction for test (e.g. 15%), set to 0 for final model
NUM_FEAT = 50

ENHANCED_SET = os.path.join(os.getcwd(), "intermediate", "feature_extraction_with_fs")
EXTRACTED_SET = os.path.join(os.getcwd(), "intermediate", "feature_extraction_without_fs")  # no feat gen or selection

# Training configs
K_FOLDS = 5  # n_splits KFold param
CV_FOLDS = 3  # cv=n Regressor param

# Pipeline Configuration
PREPROCESS = False
LINEAR_REGRESSION = False
RANDOM_FOREST = False
EXTRA_TREES = False
LIGHTGBM = False
CATBOOST = False  # run tests
XGBOOST = False
MULTILAYER_PERCEPTRON = False
KERAS = False  # run tests pls

TRAIN_NEW_MODELS = False  # k-fold metrics will not update unless training new models
OPEN_COMPARATOR = True  # to run, put 'streamlit run main.py' into the command line


# TODO: optuna, genetic algorithms
#  automate analysis (and retrain) TODO:  perhaps add a ml-based denoiser: If you have a large dataset of “clean” vs.
#   “noisy” knocks, you can train a deep neural network (e.g.,
#    a simple U-Net or fully connected model in the spectrogram domain) to learn a mapping from noisy signals to
#    clean signals. This is more complex to implement but can yield superior results if you have enough data and
#    consistent noise conditions. perhaps add something that exclusively pulls validation and test from lab dataset
#    perhaps combine datasets AFTER noise reduction instead (as it may differ per dataset) perhaps add normalization
#    preprocessing step after noise reductions on separate sets perhaps introduce scaling/normalization to error
#    metrics for intuition

def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rf_pipeline, et_pipeline, lgbm_pipeline, cat_pipeline, xgb_pipeline, mlp_pipeline, krs_pipeline = instantiate_pipelines()

    if TRAIN_NEW_MODELS:
        run_num = determine_run_num()
        functions.green_print(f"[RN] This is run number {run_num}")

    if PREPROCESS:
        ''' (separate into convert qilin, main body preprocess, split dataset)
        - Converts qilin .m4a files to .wav
        - Standardizes to 1 sec, 16kHz sample rate, mono-channel, 16-bit PCM encoding
        - Merges selected datasets (USE_QILIN and/or USE_LAB)
        - Creates new dataset for each noise reduction technique
        - Normalizes de-noised dataset with pydub.effects
        - Creates new dataset for each feature extraction technique
        - Generates features by yeo-johnson power transformer, robust scaling, 2nd polynomials interaction
        - Selects features with a f_regressor SelectKBest using k=50
        - Splits each dataset combination into train/test sets by TEST_SPLIT_RATIO
        '''
        pipeline_pp.main(USE_QILIN, USE_LAB, TEST_SPLIT_RATIO, NUM_FEAT)

    if LINEAR_REGRESSION:
        '''
        - Creates a linear regression model for each regularization technique
        - Trains a model for each combination of noise reduction, feature extraction, regularization
        - Evaluates each model w MAE, MSE, RMSE, R^2 on both holdout and k-fold validation datasets
        - Plots residuals and actual v predicted graphs
        '''
        pipeline_lr.proceed(TRAIN_NEW_MODELS)

    if RANDOM_FOREST:
        if TRAIN_NEW_MODELS:
            rf_pipeline.train()
        rf_pipeline.test(run_num)

    if EXTRA_TREES:
        if TRAIN_NEW_MODELS:
            et_pipeline.train()
        et_pipeline.test(run_num)

    if LIGHTGBM:
        if TRAIN_NEW_MODELS:
            lgbm_pipeline.train()
        lgbm_pipeline.test(run_num)

    if CATBOOST:
        if TRAIN_NEW_MODELS:
            cat_pipeline.train()
        cat_pipeline.test(run_num)

    if XGBOOST:
        if TRAIN_NEW_MODELS:
            xgb_pipeline.train()
        xgb_pipeline.test(run_num)

    if MULTILAYER_PERCEPTRON:
        if TRAIN_NEW_MODELS:
            mlp_pipeline.train()
        mlp_pipeline.test(run_num)

    # if KERAS:
    #     if TRAIN_NEW_MODELS:
    #         krs_pipeline.train()
    #     krs_pipeline.test(run_num)

    if OPEN_COMPARATOR:
        print("opening comparator")
        model_comparator.run_model_comparison_table()


def instantiate_pipelines():
    rf_pipeline = ModelPipeline(
        model_tag="rf", model_cls=RandomForestRegressor(random_state=42, n_jobs=-1, verbose=0),
        # primer_functions=[primers.remove_outliers_fit, primers.quantile_transform_fit],
        inner_folds=CV_FOLDS, outer_folds=K_FOLDS,
        ht_options=["default", "grid", "random"],
        params_grid=params.rf_grid,
        params_random=params.rf_random,
        params_optuna=params.rf_optuna,
        dataset_path=ENHANCED_SET,
    )
    et_pipeline = ModelPipeline(
        model_tag="et", model_cls=ExtraTreesRegressor(random_state=42, n_jobs=-1, verbose=0),
        # primer_functions=[primers.remove_outliers_fit, primers.quantile_transform_fit],
        inner_folds=CV_FOLDS, outer_folds=K_FOLDS,
        ht_options=["default", "grid", "random"],
        params_grid=params.et_grid,
        params_random=params.et_random,
        params_optuna=params.et_optuna,
        dataset_path=ENHANCED_SET
    )
    lgbm_pipeline = ModelPipeline(
        model_tag="lgbm", model_cls=lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        # primer_functions=[primers.log_transform, primers.standard_scale_fit],
        inner_folds=CV_FOLDS, outer_folds=K_FOLDS, early_stopping_rounds=10, use_eval_set=True,  # does not support early stopping
        ht_options=["default", "grid", "random", "bayesian"],
        params_grid=params.lgbm_grid,
        params_random=params.lgbm_random,
        params_bayesian=params.lgbm_bayesian,
        params_optuna=params.lgbm_optuna,
        dataset_path=ENHANCED_SET
    )
    cat_pipeline = ModelPipeline(
        model_tag="cat", model_cls=CatBoostRegressor(random_state=42, silent=True),
        # primer_functions=[primers.log_transform],
        inner_folds=CV_FOLDS, outer_folds=K_FOLDS, early_stopping_rounds=10, use_eval_set=True,
        ht_options=["default", "grid", "random", "bayesian"],
        params_grid=params.cat_grid,
        params_random=params.cat_random,
        params_bayesian=params.cat_bayesian,
        params_optuna=params.cat_optuna,
        dataset_path=EXTRACTED_SET  # no feature generation and selection
    )
    xgb_pipeline = ModelPipeline(
        model_tag="xgb", model_cls=XGBRegressor(random_state=42, eval_metric='rmse', verbosity=0),
        # primer_functions=[primers.log_transform, primers.standard_scale_fit],
        inner_folds=CV_FOLDS, outer_folds=K_FOLDS, early_stopping_rounds=10, use_eval_set=True,
        ht_options=["default", "grid", "random", "bayesian"],
        params_grid=params.xgb_grid,
        params_random=params.xgb_random,
        params_bayesian=params.xgb_bayesian,
        params_optuna=params.xgb_optuna,
        dataset_path=ENHANCED_SET
    )

    mlp_pipeline = ModelPipeline(
        model_tag="mlp", model_cls=MLPWrapper(verbose=0),
        inner_folds=CV_FOLDS, outer_folds=K_FOLDS,  # early stopping set to true in wrapper
        ht_options=["default", "grid", "random", "bayesian"],
        params_grid=params.mlp_grid,
        params_random=params.mlp_random,
        params_bayesian=params.mlp_bayesian,
        dataset_path=ENHANCED_SET
    )

    krs_pipeline = ModelPipeline(  # doesnt work
        model_tag="krs", model_cls=FixedKerasRegressor(
            model=build_keras_model,
            verbose=0,
            input_shape=(NUM_FEAT,),
            epochs=50,  # increase this, will make runtime worse but less error metric vals
            optimizer="adam",
            metrics=["mse"]
        ),
        inner_folds=CV_FOLDS, outer_folds=K_FOLDS, early_stopping_rounds=10,
        ht_options=["default"],  # cannot run bayesian. see comments in lines 1-2
        params_grid=params.krs_grid,
        params_random=params.krs_random,
        dataset_path=ENHANCED_SET
    )

    return rf_pipeline, et_pipeline, lgbm_pipeline, cat_pipeline, xgb_pipeline, mlp_pipeline, krs_pipeline


def build_keras_model(input_shape=None, n_layers=2, layer_size=64, dropout=0.0, optimizer="adam", **kwargs):
    """
    Build a simple Keras model for regression using a fixed number of layers
    with the same number of units in each layer.

    Parameters:
      n_layers: int
          Number of hidden layers.
      layer_size: int
          Number of neurons in each hidden layer.
      dropout: float
          Dropout rate applied after each hidden layer (if > 0).
      optimizer: str
          The optimizer to use.
      **kwargs:
          Additional keyword arguments (ignored here).

    Returns:
      A compiled Keras model.
      :param optimizer:
      :param dropout:
      :param layer_size:
      :param n_layers:
      :param input_shape:
    """
    if input_shape is None:
        raise ValueError("input_shape must be provided by scikeras based on the training data.")

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(layer_size, activation='relu'))
    for _ in range(n_layers - 1):
        model.add(Dense(layer_size, activation='relu'))
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(1))  # output layer for regression
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def determine_run_num():
    history_path = os.path.join(os.getcwd(), "output", "report_history.txt")

    # Determine next run number by reading the last run number in the file (if it exists)
    run_num = 1
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            lines = f.readlines()
            # Skip header if present (assuming header starts with "Run,")
            data_lines = [line.strip() for line in lines if line.strip() and not line.startswith("Run,")]
            if data_lines:
                try:
                    # Parse the run number of the last line
                    last_run = int(data_lines[-1].split(",")[0])
                    run_num = last_run + 1
                except Exception:
                    run_num = 1

    return run_num


class FixedKerasRegressor(KerasRegressor):
    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        # Remove keys that may cause conflicts in hyperparameter tuning
        params.pop("loss", None)
        params.pop("metrics", None)
        return params

    def _get_compile_kwargs(self):
        # Force compile options regardless of external parameters.
        return {"loss": "mean_squared_error", "metrics": ["mse"]}

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)


if __name__ == "__main__":
    main()
