# monitoring/alerts.py

import logging

logging.basicConfig(
    filename="logs/drift_alerts.log",
    level=logging.WARNING,
    format="%(asctime)s | %(message)s"
)


def send_alert(drift_report):
    """
    Alert mechanism (log-based for now)
    """
    message = (
        f"DRIFT ALERT | Feature: {drift_report['feature']} | "
        f"PSI: {drift_report['psi']:.4f} | "
        f"KS p-value: {drift_report['ks_pvalue']:.4f}"
    )

    logging.warning(message)
    print(message)
