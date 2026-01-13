import numpy as np
import os
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.npz")

def save_model(model):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # 🔴 Create temp file INSIDE artifacts folder (same drive)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".npz",
        dir=ARTIFACT_DIR
    )
    os.close(fd)  # close file descriptor (Windows requirement)

    np.savez(
        tmp_path,
        W1=model.W1,
        b1=model.b1,
        W2=model.W2,
        b2=model.b2
    )

    os.replace(tmp_path, MODEL_PATH)  # safe: same drive

def load_model(model):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train first.")

    data = np.load(MODEL_PATH)
    model.W1 = data["W1"]
    model.b1 = data["b1"]
    model.W2 = data["W2"]
    model.b2 = data["b2"]
