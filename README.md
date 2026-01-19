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

## 🔬 Research & Exploratory Analysis

The full research workflow is available in the Jupyter notebook  
[`notebook_OT.ipynb`](notebook_OT.ipynb).

It contains:

- detailed exploratory data analysis (EDA)
- data quality checks and visualizations
- leakage-safe train/test splitting
- feature selection for high-dimensional transcriptomic data
- comparison of multiple models
- hyperparameter tuning
- model evaluation and interpretation

The notebook documents the complete end-to-end analytical process that led to the final production model.


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

# 🔎 Model Verification (Live Demo)

The deployed model was verified via the FastAPI interactive interface (Swagger UI).

✅ Service health check
The /health endpoint confirms that the model artifacts are loaded correctly and the inference contract is initialized.

![4DA534F8-2FCF-4C06-9C9A-AB5AD908210B](https://github.com/user-attachments/assets/a704ae52-c87d-4404-9b0f-2b190febc74a)
![9055BE37-0F34-49AB-B838-36DA7113B105](https://github.com/user-attachments/assets/499e86d9-94ed-467f-9217-6d605cb9225a)



✅ Model inference
The /predict endpoint successfully returns the probability of operational tolerance for a valid input payload.
The same behavior is observed both in local execution and inside the Docker container.

![81B26162-B1AB-48F0-8A3F-CB745BBFF3F3](https://github.com/user-attachments/assets/6ee95a84-50ca-49de-aee7-b42e6825fd44)



# 🐳 Docker

The application is fully containerized.

```
docker build -t ot-api .
docker run --rm -p 8000:8000 ot-api

```
The API behaves identically inside and outside the container. 

![C2C71BCB-2A3F-4986-A5C5-5506A2ADD2BE](https://github.com/user-attachments/assets/5aad2a38-dedf-49a4-b58d-1dd9e54c1117)

![25B6EEF9-1398-4D16-ACBE-27F0C61B19B3](https://github.com/user-attachments/assets/7d6301fe-1901-4c33-9abe-6c117b0a0e70)


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
