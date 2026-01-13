# features/feature_definitions.py

from typing import List

# Ordered list of features expected by the model
FEATURE_COLUMNS: List[str] = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

TARGET_COLUMN = "Class"

# Simple transformation config (can grow later)
TRANSFORMS = {
    "scale_amount": True
}

MODEL_INPUT_DIM = len(FEATURE_COLUMNS)
