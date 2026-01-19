# liver-operational-tolerance_prediction

# 🧬 Operational Tolerance Prediction

Production-ready ML system for predicting immunosuppression withdrawal outcomes after liver transplantation

🔬 Overview

This project implements a production-grade machine learning system that predicts operational tolerance in liver transplant recipients using PBMC gene expression profiles.

Unlike a notebook-only solution, the model is deployed as a containerized FastAPI service with a fixed inference contract, automated tests, and reproducible environment.

🎯 Problem

Long-term immunosuppressive therapy after liver transplantation causes severe complications.
However, a subset of patients can safely discontinue immunosuppression (operational tolerance).

Goal:
Predict which patients can successfully withdraw immunosuppressive therapy using transcriptomic data.


# 📊 Data

Dataset: GEO GSE28842 
Download here: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE28nnn/GSE28842/matrix/ 

Samples: 98 liver transplant recipients

Target:

1 — Operationally tolerant

0 — Non-tolerant

# Features:

Genome-wide gene expression

Top-500 genes selected on training data only

# 🧠 Modeling

Models evaluated:

- Logistic Regression (L1)
-Random Forest
-LightGBM (final model)

#Final performance:
- ROC-AUC (test): ~0.77
Signal is multivariate and non-linear


# 🧱 Fixed Inference Contract

The production contract is explicitly enforced:

- Input: N × 500 gene expression values
-Gene order: fixed and versioned
- Preprocessing: none at inference
- Output: probability of operational tolerance
- Contract validity is verified by automated tests.

# 🚀 API Deployment (FastAPI)

The model is exposed as a REST API.

Run locally

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

# Interactive documentation
```
http://localhost:8000/docs

```

# 🐳 Docker

The application is fully containerized.

```
docker build -t ot-api .
docker run --rm -p 8000:8000 ot-api

```
The API behaves identically inside and outside the container. 

# 🧪 Testing

A contract test ensures:
- model loads correctly
- gene list integrity
- correct prediction output shape
```
pytest
```

# 🔁 Reproducibility 
- Fixed dependency versions
- Exported training pipeline
- Versioned artifacts
- Docker-based environment isolation

# Project STRUCTURE 
```
liver-tolerance-prediction/
├── app.py
├── train_model.py
├── predict.py
├── notebook_OT.ipynb
├── requirements.txt
├── Dockerfile
├── models/
│   ├── OT_LGBM_model.pkl
│   └── OT_top500_genes.csv
└── tests/
```

# ⚠️ Limitations & Future Work

- Small cohort size
- No external validation cohort
- Biological interpretation not covered

# Planned extensions:

- External validation
-Model explainability (SHAP)
- Cloud deployment (public endpoint)

# ✅ Highlights

✔ End-to-end ML pipeline
✔ Production-ready inference API
✔ Dockerized deployment
✔ Explicit inference contract
✔ Reproducible and testable

# 💼 Author’s note

This project reflects my focus on production ML systems, not just modeling — including deployment, reproducibility, and reliability.
