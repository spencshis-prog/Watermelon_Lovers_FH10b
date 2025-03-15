rf_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    # 'min_samples_leaf': [1, 2, 4],  # added
    # 'max_features': ['sqrt', 'log2', None]  # added
}

rf_random = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6],  # added
    'max_features': ['sqrt', 'log2', None]  # added
}

et_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    # 'min_samples_leaf': [1, 2, 4],  # added
    # 'max_features': ['sqrt', 'log2', None]  # added
}

et_random = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6],  # added
    'max_features': ['sqrt', 'log2', None]  # added
}

xgb_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1],
    # 'gamma': [0, 0.1, 0.5],  # determines best regularization parameter
    # 'reg_alpha': [0, 0.1, 1.0],  # determines best l1 parameter
    # 'reg_lambda': [1, 1.5, 2.0],  # determines best l2 parameter
    # 'subsample': [0.8, 1.0]  # optional
}

xgb_random = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 6, 10, 12],
    'learning_rate': [0.001, 0.01, 0.1],
    'gamma': [0, 0.1, 0.5],  # determines best regularization parameter
    'reg_alpha': [0, 0.1, 1.0],  # determines best l1 parameter
    'reg_lambda': [1, 1.5, 2.0],  # determines best l2 parameter
    'subsample': [0.6, 0.8, 1.0]
}

xgb_bayesian = {
    'n_estimators': (50, 300),
    'max_depth': (3, 12),
    'learning_rate': (1e-3, 1e-1, 'log-uniform'),
    'gamma': (0, 0.5),  # continuous between 0 and 0.5
    'reg_alpha': (0, 1.0),
    'reg_lambda': (1, 2.0),
    'subsample': (0.6, 1.0, 'uniform')
}

cat_grid = {
    'iterations': [100, 200],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1],
    # 'l2_leaf_reg': [1, 3, 5]  # added
}

cat_random = {
    'iterations': [100, 200, 300],
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.001, 0.01, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7]
}

cat_bayesian = {
    'iterations': (100, 300),
    'depth': (4, 10),
    'learning_rate': (1e-3, 1e-1, 'log-uniform'),
    'l2_leaf_reg': (1, 10)
}

lgbm_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'learning_rate': [0.01, 0.1],
    'num_leaves': [31, 63]  # default is 31 in LightGBM
}

lgbm_random = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'learning_rate': [0.001, 0.01, 0.1],
    'num_leaves': [15, 31, 63, 127],
    'min_child_samples': [5, 10, 20],
    'subsample': [0.6, 0.8, 1.0],          # corresponds to bagging_fraction
    'colsample_bytree': [0.6, 0.8, 1.0]      # fraction of features to consider at each split
}

lgbm_bayesian = {
    'n_estimators': (50, 300),
    'max_depth': (3, 30),
    'learning_rate': (1e-3, 1e-1, 'log-uniform'),
    'num_leaves': (15, 127),
    'min_child_samples': (5, 20),
    'subsample': (0.6, 1.0, 'uniform'),
    'colsample_bytree': (0.6, 1.0, 'uniform')
}

# params.py

# ----------------------------
# Optuna search spaces presets
# ----------------------------

# RandomForest / ExtraTrees: Both use similar hyperparameters.
rf_optuna = {
    'n_estimators': (50, 300),            # integer range
    'max_depth': (5, 30),                 # integer range
    'min_samples_split': (2, 10),         # integer range
    'min_samples_leaf': (1, 6),           # integer range
    # For categorical parameters, you can simply list the options.
    'max_features': ['sqrt', 'log2', None]
}
et_optuna = {
    'n_estimators': (50, 300),            # integer range
    'max_depth': (5, 30),                 # integer range
    'min_samples_split': (2, 10),         # integer range
    'min_samples_leaf': (1, 6),           # integer range
    # For categorical parameters, you can simply list the options.
    'max_features': ['sqrt', 'log2', None]
}

# XGBoost search space
xgb_optuna = {
    'n_estimators': (50, 300),
    'max_depth': (3, 12),
    'learning_rate': (1e-3, 1e-1, 'log-uniform'),  # log-uniform distribution
    'gamma': (0, 0.5),
    'reg_alpha': (0, 1.0),
    'reg_lambda': (1, 2.0),
    'subsample': (0.6, 1.0, 'uniform')
}

# LightGBM search space
lgbm_optuna = {
    'n_estimators': (50, 300),
    'max_depth': (3, 30),
    'learning_rate': (1e-3, 1e-1, 'log-uniform'),
    'num_leaves': (15, 127),            # integer range
    'min_child_samples': (5, 20),       # integer range
    'subsample': (0.6, 1.0, 'uniform'), # same as bagging_fraction
    'colsample_bytree': (0.6, 1.0, 'uniform')
}

# CatBoost search space
cat_optuna = {
    'iterations': (100, 300),
    'depth': (4, 10),
    'learning_rate': (1e-3, 1e-1, 'log-uniform'),
    'l2_leaf_reg': (1, 10)              # integer range
}

