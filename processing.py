# app/processing.py

import cv2
import numpy as np
from typing import Tuple

from .tf_models import get_crowd_model

CROWD_MODEL = None


def load_model() -> None:
    """
    Called at startup. Loads the TensorFlow crowd model.
    """
    global CROWD_MODEL
    CROWD_MODEL = get_crowd_model(
        weights_path="models/crowd_count_model.h5",
        input_shape=(64, 64, 1),
    )


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Resize, grayscale, normalize, add batch dimension.
    """
    resized = cv2.resize(frame, (64, 64))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    norm = gray.astype("float32") / 255.0
    input_tensor = np.expand_dims(norm, axis=(0, -1))  # (1,64,64,1)
    return input_tensor


def estimate_crowd_from_frame(frame: np.ndarray) -> Tuple[int, str]:
    """
    Use TF model (if loaded) to estimate people count.
    Falls back to simple edge-based heuristic if needed.
    """
    global CROWD_MODEL

    if CROWD_MODEL is not None:
        inp = preprocess_frame(frame)
        pred = CROWD_MODEL.predict(inp, verbose=0)
        estimated_people = int(float(pred[0][0]))
    else:
        # Fallback heuristic
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_count = int(np.sum(edges > 0))
        estimated_people = max(0, edge_count // 200)

    if estimated_people < 20:
        density_level = "low"
    elif estimated_people < 50:
        density_level = "medium"
    elif estimated_people < 100:
        density_level = "high"
    else:
        density_level = "critical"

    return estimated_people, density_level