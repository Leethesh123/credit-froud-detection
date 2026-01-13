# monitoring/drift.py

import json
import numpy as np
from scipy.stats import ks_2samp

REFERENCE_PATH = "monitoring/reference_stats.json"

PSI_THRESHOLD = 0.2
KS_PVALUE_THRESHOLD = 0.05


def calculate_psi(expected, actual, bins=10):
    """
    Population Stability Index
    """
    expected_hist, bin_edges = np.histogram(expected, bins=bins)
    actual_hist, _ = np.histogram(actual, bins=bin_edges)

    expected_perc = expected_hist / len(expected)
    actual_perc = actual_hist / len(actual)

    psi = np.sum(
        (actual_perc - expected_perc)
        * np.log((actual_perc + 1e-6) / (expected_perc + 1e-6))
    )
    return psi


def load_reference():
    with open(REFERENCE_PATH) as f:
        return json.load(f)


def detect_drift(feature_name, reference_values, current_values):
    """
    Returns drift metrics for one feature
    """
    psi = calculate_psi(reference_values, current_values)
    ks_stat, p_value = ks_2samp(reference_values, current_values)

    drifted = psi > PSI_THRESHOLD or p_value < KS_PVALUE_THRESHOLD

    return {
        "feature": feature_name,
        "psi": float(psi),
        "ks_pvalue": float(p_value),
        "drift_detected": drifted
    }
