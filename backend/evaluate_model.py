"""Evaluate the AI Job Risk models and print accuracy, precision, recall, F1 and ROC-AUC.

Place this file inside the repo's backend/ folder and run:  python evaluate_model.py
It reproduces the exact split used in train_model.py (80/20, stratified, seed 42)
and does NOT overwrite the saved model.
"""
import os
import warnings

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(HERE, "ai_impact_jobs_2010_2025.csv"))
df["target"] = (df["automation_risk_score"] >= 0.7).astype(int)

features = ["industry", "seniority_level", "company_size", "ai_intensity_score",
            "industry_ai_adoption_stage", "job_description_embedding_cluster"]
cat = ["industry", "seniority_level", "company_size",
       "industry_ai_adoption_stage", "job_description_embedding_cluster"]
num = ["ai_intensity_score"]

X, y = df[features].copy(), df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Train: {len(X_train)}  Test: {len(X_test)}  "
      f"High-risk share: {y.mean():.3f}")


def report(name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"Confusion matrix  TN={tn} FP={fp} FN={fn} TP={tp}")


# 1) Logistic Regression baseline (same as train_model.py)
pre = ColumnTransformer([("num", StandardScaler(), num),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
lr = Pipeline([("preprocessor", pre),
               ("classifier", LogisticRegression(max_iter=1000, random_state=42,
                                                 class_weight="balanced"))])
lr.fit(X_train, y_train)
report("Logistic Regression (baseline)", y_test, lr.predict_proba(X_test)[:, 1])

# 2) CatBoost, same configuration family as train_model.py
Xtr, Xte = X_train.copy(), X_test.copy()
for c in cat:
    Xtr[c], Xte[c] = Xtr[c].astype(str), Xte[c].astype(str)
cb = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=5,
                        eval_metric="AUC", random_seed=42, verbose=False,
                        cat_features=cat, auto_class_weights="Balanced")
cb.fit(Xtr, y_train)
report("CatBoost (all features, depth 6, lr 0.05, l2 5)", y_test,
       cb.predict_proba(Xte)[:, 1])

# 3) The model actually saved in the repo (the one the web app uses)
path = os.path.join(HERE, "model", "final_ai_job_replacement_model.pkl")
saved = joblib.load(path)
print(f"\nSaved model type: {type(saved).__name__}")
if hasattr(saved, "feature_names_"):
    cols = list(saved.feature_names_)
    print("Saved model features:", cols)
    Xs = X_test[cols].copy()
    for c in cols:
        if c in cat:
            Xs[c] = Xs[c].astype(str)
    report("SAVED MODEL (used by the app)", y_test, saved.predict_proba(Xs)[:, 1])
else:
    report("SAVED MODEL (used by the app)", y_test, saved.predict_proba(X_test)[:, 1])
