# mlops/retrain.py

import pandas as pd
import numpy as np
from datetime import datetime

from src.model import NeuralNetwork
from src.preprocess import preprocess
from src.utils import save_model

DATA_PATH = "data/creditcard.csv"

def retrain():
    print("Starting retraining job...")

    df = pd.read_csv(DATA_PATH)
    X, y = preprocess(df)

    model = NeuralNetwork(input_dim=X.shape[1])

    model.train(
        X,
        y,
        epochs=50,
        lr=0.01,
        class_weight=True
    )

    save_model(model)

    print("Retraining completed.")
    print(f"Candidate model trained at {datetime.utcnow().isoformat()}")

if __name__ == "__main__":
    retrain()
