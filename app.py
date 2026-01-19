from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(title="Operational Tolerance Predictor")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "OT_LGBM_model.pkl"
GENES_TXT_PATH = MODELS_DIR / "OT_top500_genes.txt"
GENES_CSV_PATH = MODELS_DIR / "OT_top500_genes.csv"


def load_genes() -> list[str]:
    if GENES_TXT_PATH.exists():
        genes = [line.strip() for line in GENES_TXT_PATH.read_text().splitlines() if line.strip()]
        return genes

    if GENES_CSV_PATH.exists():
        # ожидаем один столбец с именами генов/проб
        s = pd.read_csv(GENES_CSV_PATH, header=None).iloc[:, 0].astype(str)
        genes = [g.strip() for g in s.tolist() if str(g).strip()]
        return genes

    raise FileNotFoundError(
        f"Gene list not found. Expected one of:\n"
        f"- {GENES_TXT_PATH}\n"
        f"- {GENES_CSV_PATH}\n"
    )


# Load artifacts at startup
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
genes = load_genes()

if len(genes) != 500:
    raise ValueError(f"Expected 500 genes, found {len(genes)}. Check gene file content/order.")


class PredictRequest(BaseModel):
    samples: List[List[float]] = Field(..., description="List of samples, each sample is a list of 500 gene expression values.")


@app.get("/health")
def health():
    return {"status": "ok", "n_genes": len(genes)}


@app.post("/predict")
def predict(req: PredictRequest):
    # Validate each sample length
    for i, s in enumerate(req.samples):
        if len(s) != len(genes):
            raise ValueError(f"Sample {i} has {len(s)} features, expected {len(genes)}")

    X = pd.DataFrame(req.samples, columns=genes)
    if X.isnull().any().any():
        raise ValueError("Input contains NaN values")

    proba = model.predict_proba(X)[:, 1]
    return {"predictions": proba.tolist()}

