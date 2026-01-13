import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
import os

from src.model import NeuralNetwork
from src.preprocess import preprocess
from src.utils import save_model


def train(
    data_path="data/creditcard.csv",
    epochs=50,
    threshold=0.3,
    test_size=0.2,
    random_state=42,
):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Please download it and place it in the data/ directory."
        )

    df = pd.read_csv(data_path)
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Compute positive class weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    pos_weight = neg / pos

    print("Positive class weight:", pos_weight)

    model = NeuralNetwork(
        input_dim=X_train.shape[1],
        hidden_dim=16,
        lr=0.01,
        pos_weight=pos_weight,
    )

    for epoch in range(epochs):
        y_hat = model.forward(X_train)
        loss = model.compute_loss(y_train, y_hat)
        model.backward(X_train, y_train)

        if epoch % 5 == 0:
            preds = (y_hat > threshold).astype(int)
            recall = recall_score(y_train, preds)
            precision = precision_score(y_train, preds, zero_division=0)

            print(
                f"Epoch {epoch} | Loss {loss:.4f} | "
                f"Recall {recall:.4f} | Precision {precision:.4f}"
            )

    y_test_hat = model.forward(X_test)
    test_preds = (y_test_hat > threshold).astype(int)

    print("TEST Recall:", recall_score(y_test, test_preds))
    print("TEST Precision:", precision_score(y_test, test_preds, zero_division=0))

    save_model(model)


if __name__ == "__main__":
    train()
