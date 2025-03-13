import model_comparator
from scripts.catboost import pipeline_cat
from scripts.extra_trees import pipeline_et
from scripts.lightgbm import pipeline_lgbm
from scripts.linear_regression import pipeline_lr
from scripts.preprocessing import pipeline_pp
from scripts.random_forest import pipeline_rf
from scripts.xgboost import pipeline_xgb

# Dataset Configuration
USE_QILIN = True
USE_LAB = False
USE_SEPARATE_TEST_SET = False  # set to true it we want to use all datapoints and have a distinct test set
# VAL_SPLIT_RATIO = 0.15  # fraction for validation (e.g. 15%), set to 0 for final model
TEST_SPLIT_RATIO = 0.15  # fraction for test (e.g. 15%), set to 0 for final model

# Training configs
OUTER_K_FOLDING = 5  # n_splits KFold param
INNER_K_FOLDING = 3  # cv=n Regressor param

# Pipeline Configuration
PREPROCESS = False
LINEAR_REGRESSION = False
RANDOM_FOREST = False
EXTRA_TREES = False
LIGHTGBM = False
CATBOOST = False
XGBOOST = False
NEURAL_NETWORK = False

TRAIN_NEW_MODELS = False  # k-fold metrics will not update unless training new models
OPEN_COMPARATOR = True  # to run, put 'streamlit run main.py' into the command line


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
        '''
        - Creates a random forest regressor for each hyperparameter tuning technique
        - Trains a model for each combination of noise reduction, feature extraction, regularization
        - Evaluates each model w MAE, MSE, RMSE, R^2 on both holdout and k-fold validation datasets
        - Plots residuals and actual v predicted graphs
        '''
        pipeline_rf.proceed(TRAIN_NEW_MODELS)

    if EXTRA_TREES:
        pipeline_et.proceed(TRAIN_NEW_MODELS)

    if LIGHTGBM:
        pipeline_lgbm.proceed(TRAIN_NEW_MODELS)

    if CATBOOST:
        pipeline_cat.proceed(TRAIN_NEW_MODELS)

    if XGBOOST:
        pipeline_xgb.proceed(TRAIN_NEW_MODELS)

    if OPEN_COMPARATOR:
        model_comparator.main()


if __name__ == "__main__":
    main()
