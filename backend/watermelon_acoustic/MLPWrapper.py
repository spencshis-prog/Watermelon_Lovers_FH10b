from sklearn.neural_network import MLPRegressor


class MLPWrapper:
    def __init__(self, n_layers=2, layer_size=64, activation="relu", solver="adam",
                 alpha=1e-4, learning_rate="constant", max_iter=1500, tol=1e-4,
                 early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, **kwargs):
        # Save the parameters
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.kwargs = kwargs
        self.model_ = None

    def fit(self, X, y):
        # Convert separate parameters into a hidden_layer_sizes tuple
        hidden_layer_sizes = tuple([self.layer_size] * self.n_layers)
        self.model_ = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            random_state=42,
            **self.kwargs
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def set_params(self, **params):
        # Update parameters (needed for hyperparameter tuning)
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_params(self, deep=True):
        params = {
            "n_layers": self.n_layers,
            "layer_size": self.layer_size,
            "activation": self.activation,
            "solver": self.solver,
            "alpha": self.alpha,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter
        }
        params.update(self.kwargs)
        return params
