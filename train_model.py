
# train_model.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# 0. Настройки
# -----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# 1. Загрузка данных
# -----------------------------

import GEOparse

file_path = "GSE28842_series_matrix.txt"   # ← измени на реальный путь, если нужно
# Примеры альтернативных путей:
# file_path = "datasets/GSE28842_series_matrix.txt"
# file_path = "GSE28842_series_matrix.txt"

print(f"Загружаем GEO файл: {file_path}")
gse = GEOparse.get_GEO(filepath=file_path)
expr = gse.exprs   # ← это сразу правильная DataFrame с экспрессией!

# В GEO обычно: строки = гены (пробы), столбцы = сэмплы (GSMxxxx)
# Но в твоём коде дальше ожидается ТРАНСПОНИРОВАННАЯ матрица (сэмплы в строках)
expr = expr.T      # ← очень важно! Теперь индекс — сэмплы, столбцы — гены

print("Форма матрицы экспрессии:", expr.shape)
print("Первые сэмплы:", expr.index[:5].tolist())
print("Первые гены:", expr.columns[:5].tolist())


# -----------------------------
# 2. Явное задание групп
# -----------------------------
tol_samples = [
    "GSM714219","GSM714220","GSM714221","GSM714222","GSM714223",
    "GSM714224","GSM714225","GSM714226","GSM714227","GSM714228",
    "GSM714229","GSM714230","GSM714232","GSM714233","GSM714234",
    "GSM714235","GSM714236","GSM714237","GSM714238","GSM714239"
]

nontol_samples = [
    "GSM714240","GSM714241","GSM714242","GSM714243","GSM714244",
    "GSM714245","GSM714246","GSM714247","GSM714248","GSM714249",
    "GSM714250","GSM714251","GSM714252","GSM714253","GSM714254",
    "GSM714255","GSM714256","GSM714257","GSM714258","GSM714259",
    "GSM714260","GSM714261","GSM714262","GSM714263"
]

ptol_samples = [
    "GSM714231","GSM714273","GSM714280","GSM714281","GSM714282",
    "GSM714283","GSM714284","GSM714285","GSM714286","GSM714287",
    "GSM714288","GSM714289"
]

pnontol_samples = [
    "GSM714265","GSM714266","GSM714267","GSM714268","GSM714269",
    "GSM714270","GSM714271","GSM714272","GSM714274","GSM714275",
    "GSM714276","GSM714277","GSM714278","GSM714279"
]

# -----------------------------
# 3. Создание меток
# -----------------------------
label_map = {s: 1 for s in tol_samples + ptol_samples}
label_map.update({s: 0 for s in nontol_samples + pnontol_samples})

labeled_samples = sorted([s for s in expr.columns if s in label_map])
expr_labeled = expr[labeled_samples].T
y = pd.Series({s: label_map[s] for s in labeled_samples}, name="status")

# -----------------------------
# 4. Разделение на train/test
# -----------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    expr_labeled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

assert set(X_train_raw.index).isdisjoint(X_test_raw.index)
assert X_train_raw.shape[0] + X_test_raw.shape[0] == expr_labeled.shape[0]

# -----------------------------
# 5. Feature scaling (для модели не нужен в inference)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# -----------------------------
# 6. Модели и подбор гиперпараметров
# -----------------------------
models = {
    "LGBM": LGBMClassifier(random_state=RANDOM_STATE)
}

param_grids = {
    "LGBM": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [-1],
        "num_leaves": [31, 50]
    }
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
best_models = {}

for name, model in models.items():
    grid = GridSearchCV(model, param_grids[name], scoring="roc_auc", cv=cv, n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    best_models[name] = grid.best_estimator_
    print(f"{name} best params: {grid.best_params_}, best CV ROC-AUC: {grid.best_score_:.3f}")

# Выбираем LGBM как финальную модель
best_lgbm = best_models["LGBM"]

# -----------------------------
# 7. Оценка на тесте
# -----------------------------
test_pred = best_lgbm.predict_proba(X_test_scaled)[:,1]
test_auc = roc_auc_score(y_test, test_pred)
print(f"Test ROC-AUC: {test_auc:.3f}")

# -----------------------------
# 8. Сохранение модели и топ-500 генов
# -----------------------------
# Выбираем top-500 генов по variance
top500_genes = X_train_raw.var(axis=0).sort_values(ascending=False).index[:500].tolist()

joblib.dump(best_lgbm, os.path.join(MODEL_DIR, "OT_LGBM_model.pkl"))

with open(os.path.join(MODEL_DIR, "OT_top500_genes.txt"), "w") as f:
    for g in top500_genes:
        f.write(g + "\n")

print("Model and top-500 genes saved!")
