import pipeline_lr
import pipeline_pp

# Configuration booleans
USE_QILIN = True
USE_LAB = False

USE_SEPARATE_TEST_SET = False  # set to true it we want to use all datapoints and have a distinct test set
# VAL_SPLIT_RATIO = 0.15  # fraction for validation (e.g. 15%), set to 0 for final model
TEST_SPLIT_RATIO = 0.15  # fraction for test (e.g. 15%), set to 0 for final model

# major pipeline stages
PREPROCESS = True
LINEAR_REGRESSION = True
RANDOM_FOREST = False
XGBOOST = False
NEURAL_NETWORK = False


# TODO: combined testing will try every dataset-noise reduction combination perhaps eliminates dataset-
# TODO: dimension below unless one dataset strictly worsens the medley
# TODO: sort both kfold and holdout report entry orders so they are easily comparable
# perhaps add something that exclusively pulls validation and test from lab dataset
# perhaps combine datasets AFTER noise reduction instead (as it may differ per dataset)

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
        pipeline_lr.proceed()


if __name__ == "__main__":
    main()
