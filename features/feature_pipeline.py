# features/feature_pipeline.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from features.feature_definitions import FEATURE_COLUMNS, TARGET_COLUMN, TRANSFORMS

# Global scaler (fitted during training, reused in inference if persisted)
_scaler = StandardScaler()

def fit_transform(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Used during training.
    Fits transformations and returns X, y.
    """
    X = df[FEATURE_COLUMNS].copy()

    if TRANSFORMS.get("scale_amount", False):
        X["Amount"] = _scaler.fit_transform(X[["Amount"]])

    y = df[TARGET_COLUMN].values
    return X.values, y


def transform(df: pd.DataFrame) -> np.ndarray:
    """
    Used during inference.
    Applies SAME transformations as training.
    """
    X = df[FEATURE_COLUMNS].copy()

    if TRANSFORMS.get("scale_amount", False):
        X["Amount"] = _scaler.transform(X[["Amount"]])

    return X.values
