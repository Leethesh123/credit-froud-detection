from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os, sys

sys.path.append(os.path.abspath("."))

from src.model import NeuralNetwork
from src.utils import load_model

app = FastAPI(title="Credit Card Fraud Detection API")

# -------- Request Schema --------
class PredictionRequest(BaseModel):
    features: list[float]

# -------- Load Model --------
model = NeuralNetwork(input_dim=30)

try:
    load_model(model)
    model_loaded = True
except Exception as e:
    print("Model load failed:", e)
    model_loaded = False

# -------- Health Check --------
@app.get("/")
def health():
    return {
        "status": "running",
        "model_loaded": model_loaded
    }

# -------- Prediction Endpoint --------
@app.post("/predict")
def predict(request: PredictionRequest):
    if not model_loaded:
        return {"error": "Model not loaded. Train model first."}

    if len(request.features) != 30:
        return {"error": "features must contain exactly 30 values"}

    X = np.array(request.features).reshape(1, -1)
    prob = float(model.forward(X)[0][0])
    prediction = int(prob > 0.3)

    return {
        "fraud_probability": prob,
        "is_fraud": prediction
    }
