# mlops/evaluate.py

import pandas as pd
from sklearn.metrics import recall_score, precision_score

from src.model import NeuralNetwork
from src.preprocess import preprocess
from src.utils import load_model

DATA_PATH = "data/creditcard.csv"

RECALL_THRESHOLD = 0.95   # business rule
PRECISION_THRESHOLD = 0.001

def evaluate():
    print("Evaluating candidate model...")

    df = pd.read_csv(DATA_PATH)
    X, y = preprocess(df)

    model = NeuralNetwork(input_dim=X.shape[1])
    load_model(model)

    probs = model.forward(X).flatten()
    preds = (probs > 0.3).astype(int)

    recall = recall_score(y, preds)
    precision = precision_score(y, preds)

    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")

    if recall >= RECALL_THRESHOLD:
        print("Model PASSED evaluation")
        return True
    else:
        print("Model FAILED evaluation")
        return False

if __name__ == "__main__":
    evaluate()
