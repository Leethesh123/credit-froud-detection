# mlops/promote_model.py

from mlops.evaluate import evaluate
import shutil
import os

CANDIDATE_MODEL = "artifacts/model.npz"
PRODUCTION_MODEL = "artifacts/production_model.npz"

def promote():
    print("Promotion check started...")

    passed = evaluate()

    if not passed:
        print("Model not promoted.")
        return

    if not os.path.exists(CANDIDATE_MODEL):
        raise FileNotFoundError("Candidate model not found")

    shutil.copyfile(CANDIDATE_MODEL, PRODUCTION_MODEL)

    print("Model PROMOTED to production.")

if __name__ == "__main__":
    promote()
