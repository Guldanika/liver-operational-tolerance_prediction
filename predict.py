from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "OT_LGBM_model.pkl"
GENES_CSV_PATH = MODELS_DIR / "OT_top500_genes.csv"
GENES_TXT_PATH = MODELS_DIR / "OT_top500_genes.txt"

model = joblib.load(MODEL_PATH)

if GENES_TXT_PATH.exists():
    genes = [line.strip() for line in GENES_TXT_PATH.read_text().splitlines() if line.strip()]
else:
    genes = pd.read_csv(GENES_CSV_PATH, header=None).iloc[:, 0].astype(str).str.strip().tolist()

X = pd.DataFrame([[0.0] * len(genes)], columns=genes)
proba = model.predict_proba(X)[0, 1]
print(f"Predicted probability of operational tolerance (TOL): {proba:.4f}")



