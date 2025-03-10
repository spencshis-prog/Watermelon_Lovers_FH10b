import tensorflow as tf


def build_lr_model(input_shape, regularization="none"):
    """
    Builds a simple linear regression model (i.e. one Dense layer, no hidden layers)
    with optional regularization.
    regularization: one of "none", "lasso", "ridge", "ElasticNet".
    """
    if regularization == "none":
        reg = None
    elif regularization == "lasso":
        reg = tf.keras.regularizers.l1(0.01)
    elif regularization == "ridge":
        reg = tf.keras.regularizers.l2(0.01)
    elif regularization == "ElasticNet":
        reg = tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    else:
        reg = None

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense

    model = Sequential([
        Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=reg)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# Export the list of regularization options.
REG_OPTIONS = ["none", "lasso", "ridge", "ElasticNet"]
