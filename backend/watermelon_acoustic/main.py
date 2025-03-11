import model_comparator
from scripts_linear_regression import pipeline_lr
from scripts_preprocessing import pipeline_pp
from scripts_random_forest import pipeline_rf
from scripts_xgboost import pipeline_xgb

# Dataset Configuration
USE_QILIN = True
USE_LAB = False
USE_SEPARATE_TEST_SET = False  # set to true it we want to use all datapoints and have a distinct test set
# VAL_SPLIT_RATIO = 0.15  # fraction for validation (e.g. 15%), set to 0 for final model
TEST_SPLIT_RATIO = 0.15  # fraction for test (e.g. 15%), set to 0 for final model

# Pipeline Configuration
PREPROCESS = False
LINEAR_REGRESSION = False
RANDOM_FOREST = False
XGBOOST = False
NEURAL_NETWORK = False

TRAIN_NEW_MODELS = False  # k-fold metrics will not update unless training new models
OPEN_COMPARATOR = True  # to run, put 'streamlit run main.py' into the command line


# TODO: combined testing will try every dataset-noise reduction combination perhaps eliminates dataset-
# TODO: dimension below unless one dataset strictly worsens the medley
# perhaps add something that exclusively pulls validation and test from lab dataset
# perhaps combine datasets AFTER noise reduction instead (as it may differ per dataset)
# perhaps add normalization preprocessing step after noise reductions on separate sets

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

    if XGBOOST:
        pipeline_xgb.proceed(TRAIN_NEW_MODELS)

    if OPEN_COMPARATOR:
        model_comparator.main()


if __name__ == "__main__":
    main()
