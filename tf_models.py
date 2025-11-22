# app/tf_models.py

import os
from typing import Tuple

import tensorflow as tf
# expose keras submodules from the tf alias so static analyzers that can't resolve
# "tensorflow.keras" will still find layers and models.
layers = tf.keras.layers
models = tf.keras.models


# ============
# 1. CROWD MODEL (IMAGE -> PEOPLE COUNT)
# ============

def build_crowd_model(input_shape: Tuple[int, int, int] = (64, 64, 1)) -> tf.keras.Model:
    """
    Simple CNN to estimate number of people from an image.
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="relu")(x)  # non-negative count

    model = models.Model(inputs=inputs, outputs=outputs, name="crowd_count_model")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def get_crowd_model(
    weights_path: str = "models/crowd_count_model.h5",
    input_shape: Tuple[int, int, int] = (64, 64, 1),
) -> tf.keras.Model:
    """
    Load model if weights exist; otherwise create a new untrained model.
    """
    model = build_crowd_model(input_shape=input_shape)
    if os.path.exists(weights_path):
        print(f"[TF] Loading crowd model weights from {weights_path}")
        model.load_weights(weights_path)
    else:
        print(f"[TF] No crowd model weights at {weights_path}. Using random init.")
    return model


# ============
# 2. RISK MODEL (TABULAR FEATURES -> RISK SCORE)
# ============

def build_risk_model(num_features: int) -> tf.keras.Model:
    """
    Simple MLP: [people, queue, temp, density_score, ...] -> risk in [0,1]
    """
    inputs = layers.Input(shape=(num_features,))

    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # 0-1 risk

    model = models.Model(inputs=inputs, outputs=outputs, name="risk_score_model")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def get_risk_model(
    num_features: int,
    weights_path: str = "models/risk_score_model.h5",
) -> tf.keras.Model:
    model = build_risk_model(num_features)
    if os.path.exists(weights_path):
        print(f"[TF] Loading risk model weights from {weights_path}")
        model.load_weights(weights_path)
    else:
        print(f"[TF] No risk model weights at {weights_path}. Using random init.")
    return model