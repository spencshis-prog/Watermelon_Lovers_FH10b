from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from skopt.learning import RandomForestRegressor
from xgboost import XGBRegressor

import model_comparator
import params
import primers
from scripts.Pipeline.ModelPipeline import ModelPipeline
from scripts.linear_regression import pipeline_lr
from scripts.preprocessing import pipeline_pp


import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore", message="n_quantiles.*")


# Dataset Configuration
USE_QILIN = True
USE_LAB = False
USE_SEPARATE_TEST_SET = False  # set to true it we want to use all datapoints and have a distinct test set
# VAL_SPLIT_RATIO = 0.15  # fraction for validation (e.g. 15%), set to 0 for final model
TEST_SPLIT_RATIO = 0.15  # fraction for test (e.g. 15%), set to 0 for final model

# Training configs
K_FOLDS = 5  # n_splits KFold param
CV_FOLDS = 3  # cv=n Regressor param

# Pipeline Configuration
PREPROCESS = False
LINEAR_REGRESSION = False
RANDOM_FOREST = True
EXTRA_TREES = False
LIGHTGBM = False
CATBOOST = False  # run tests
XGBOOST = False
NEURAL_NETWORK = False

TRAIN_NEW_MODELS = True  # k-fold metrics will not update unless training new models
OPEN_COMPARATOR = False  # to run, put 'streamlit run main.py' into the command line


# TODO: LightGDB, optuna, genetic algorithms
# TODO: data dump for error metrics for each hypertuning combination, automate analysis (and retrain)
# TODO: combined testing will try every dataset-noise reduction combination perhaps eliminates dataset-
# TODO: dimension below unless one dataset strictly worsens the medley
# perhaps add a ml-based denoiser: If you have a large dataset of “clean” vs. “noisy” knocks, you can train a deep neural network (e.g., a simple U-Net or fully connected model in the spectrogram domain) to learn a mapping from noisy signals to clean signals. This is more complex to implement but can yield superior results if you have enough data and consistent noise conditions.
# perhaps add something that exclusively pulls validation and test from lab dataset
# perhaps combine datasets AFTER noise reduction instead (as it may differ per dataset)
# perhaps add normalization preprocessing step after noise reductions on separate sets
# perhaps introduce scaling/normalization to error metrics for intuition

# concern: qilin dataset seems to range between 9-12 sweetness, never lower. top-heavy training set
def main():
    rf_pipeline, et_pipeline, lgbm_pipeline, cat_pipeline, xgb_pipeline = instantiate_pipelines()

    if PREPROCESS:
        '''
        - Converts qilin .m4a files to .wav
        - Standardizes to 1 sec, 16kHz sample rate, mono-channel, 16-bit PCM encoding
        - Merges selected datasets (USE_QILIN and/or USE_LAB)
        - Creates new dataset for each noise reduction technique
        - Creates new dataset for each feature extraction technique
        - Splits each dataset combination into train/test sets by TEST_SPLIT_RATIO
        '''
        pipeline_pp.proceed(USE_QILIN, USE_LAB, USE_SEPARATE_TEST_SET, TEST_SPLIT_RATIO)

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
        rf_pipeline.test()

    if EXTRA_TREES:
        if TRAIN_NEW_MODELS:
            et_pipeline.train()
        et_pipeline.test()

    if LIGHTGBM:
        if TRAIN_NEW_MODELS:
            lgbm_pipeline.train()
        lgbm_pipeline.test()

    if CATBOOST:
        if TRAIN_NEW_MODELS:
            cat_pipeline.train()
        cat_pipeline.test()

    if XGBOOST:
        if TRAIN_NEW_MODELS:
            xgb_pipeline.train()
        xgb_pipeline.test()

    if OPEN_COMPARATOR:
        print("opening comparator")
        model_comparator.main()


def instantiate_pipelines():
    rf_pipeline = ModelPipeline(
        model_tag="rf", model_cls=RandomForestRegressor(random_state=42, n_jobs=-1, verbose=0),
        primer_functions=None,  # [primers.remove_outliers, primers.quantile_transform],
        inner_folds=CV_FOLDS, outer_folds=K_FOLDS,
        ht_options=["default", "grid", "random"],
        params_grid=params.rf_grid,
        params_random=params.rf_random,
        params_optuna=params.rf_optuna
    )
    et_pipeline = ModelPipeline(
        model_tag="et", model_cls=ExtraTreesRegressor(random_state=42, n_jobs=-1, verbose=0),
        primer_functions=[primers.remove_outliers, primers.quantile_transform],
        inner_folds=CV_FOLDS, outer_folds=K_FOLDS,
        ht_options=["default", "grid", "random"],
        params_grid=params.et_grid,
        params_random=params.et_random,
        params_optuna=params.et_optuna
    )
    lgbm_pipeline = ModelPipeline(
        model_tag="lgbm", model_cls=lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        primer_functions=[primers.log_transform, primers.standard_scale],
        inner_folds=CV_FOLDS, outer_folds=K_FOLDS,
        ht_options=["default", "grid", "random", "bayesian"],
        params_grid=params.lgbm_grid,
        params_random=params.lgbm_random,
        params_bayesian=params.lgbm_bayesian,
        params_optuna=params.lgbm_optuna
    )
    cat_pipeline = ModelPipeline(
        model_tag="cat", model_cls=CatBoostRegressor(random_state=42, silent=True, verbose=0),
        primer_functions=[primers.log_transform],
        inner_folds=CV_FOLDS, outer_folds=K_FOLDS,
        ht_options=["default", "grid", "random", "bayesian"],
        params_grid=params.cat_grid,
        params_random=params.cat_random,
        params_bayesian=params.cat_bayesian,
        params_optuna=params.cat_optuna
    )
    xgb_pipeline = ModelPipeline(
        model_tag="xgb", model_cls=XGBRegressor(random_state=42, eval_metric='rmse', verbosity=0),
        primer_functions=[primers.log_transform, primers.standard_scale],
        inner_folds=CV_FOLDS, outer_folds=K_FOLDS,
        ht_options=["default", "grid", "random", "bayesian"],
        params_grid=params.xgb_grid,
        params_random=params.xgb_random,
        params_bayesian=params.xgb_bayesian,
        params_optuna=params.xgb_optuna
    )

    return rf_pipeline, et_pipeline, lgbm_pipeline, cat_pipeline, xgb_pipeline

    # lgbm_pipeline.train()
    # lgbm_pipeline.test()


if __name__ == "__main__":
    main()
