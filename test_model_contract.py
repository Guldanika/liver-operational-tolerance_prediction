from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]   # корень проекта
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "OT_LGBM_model.pkl"
GENES_TXT_PATH = MODELS_DIR / "OT_top500_genes.txt"
GENES_CSV_PATH = MODELS_DIR / "OT_top500_genes.csv"


def load_genes() -> list[str]:
    # 1) Prefer TXT if exists
    if GENES_TXT_PATH.exists():
        genes = [line.strip() for line in GENES_TXT_PATH.read_text().splitlines() if line.strip()]
        return genes

    # 2) Fallback to CSV (one column with gene/probe ids)
    if GENES_CSV_PATH.exists():
        s = pd.read_csv(GENES_CSV_PATH, header=None).iloc[:, 0].astype(str)
        genes = [g.strip() for g in s.tolist() if str(g).strip()]
        return genes

    raise FileNotFoundError(
        f"Gene list not found. Expected one of:\n"
        f"- {GENES_TXT_PATH}\n"
        f"- {GENES_CSV_PATH}"
    )


def test_model_contract():
    # --- Existence checks ---
    assert MODEL_PATH.exists(), f"Model file not found: {MODEL_PATH}"

    genes = load_genes()
    assert len(genes) == 500, f"Expected 500 genes, found {len(genes)}"

    # --- Load model ---
    model = joblib.load(MODEL_PATH)

    # --- Dummy input strictly following contract (1 x 500) ---
    X = pd.DataFrame([[0.0] * len(genes)], columns=genes)

    # --- Sanity checks ---
    assert X.shape == (1, 500)
    assert not X.isnull().any().any(), "Input contains NaN values"

    # --- Inference contract: predict_proba returns (n_samples, 2) ---
    pred = model.predict_proba(X)
    assert pred.shape == (1, 2), f"Expected (1,2), got {pred.shape}"





