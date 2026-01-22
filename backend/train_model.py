import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from catboost import CatBoostClassifier

# --- MODEL LOCKING ---
FINAL_MODEL_PATH = 'final_ai_job_replacement_model.pkl'

if os.path.exists(FINAL_MODEL_PATH):
    print(f"Trained model found. Loading from disk. No retraining required.")
    exit()

# Load Data
df = pd.read_csv('ai_impact_jobs_2010_2025.csv')

# Target Definition
df['target'] = (df['automation_risk_score'] >= 0.7).astype(int)

# Features
features = [
    'industry', 
    'seniority_level', 
    'company_size', 
    'ai_intensity_score', 
    'industry_ai_adoption_stage', 
    'job_description_embedding_cluster'
]

X = df[features].copy()
y = df['target']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# STEP 1 – Baseline Verification
categorical_features = ['industry', 'seniority_level', 'company_size', 'industry_ai_adoption_stage', 'job_description_embedding_cluster']
numerical_features = ['ai_intensity_score']

lr_preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

lr_pipeline = Pipeline([
    ('preprocessor', lr_preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
])

lr_pipeline.fit(X_train, y_train)
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]
baseline_auc = roc_auc_score(y_test, y_prob_lr)

print(f"=== BASELINE MODEL (LOGISTIC REGRESSION) ===")
print(f"ROC-AUC: {baseline_auc:.4f}")

# STEP 2 – Feature Signal Refinement
# Run a quick CatBoost to get feature importance for filtering
X_train_cb = X_train.copy()
X_test_cb = X_test.copy()
for col in categorical_features:
    X_train_cb[col] = X_train_cb[col].astype(str)
    X_test_cb[col] = X_test_cb[col].astype(str)

temp_cb = CatBoostClassifier(iterations=100, verbose=False, cat_features=categorical_features, random_seed=42)
temp_cb.fit(X_train_cb, y_train)
importances = temp_cb.get_feature_importance()
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

retained_features = feat_imp[feat_imp >= 2.0].index.tolist()
removed_features = feat_imp[feat_imp < 2.0].index.tolist()

print(f"\n=== FEATURE REFINEMENT ===")
print(f"Retained: {retained_features}")
print(f"Removed: {removed_features}")

X_train_final = X_train_cb[retained_features]
X_test_final = X_test_cb[retained_features]
final_cat_features = [f for f in retained_features if f in categorical_features]

# STEP 3 – CatBoost AUC Optimization
results = []
# Reduced search space for speed
depths = [4, 6]
lrs = [0.05, 0.1]
regs = [5, 10]

print(f"\n=== STARTING CATBOOST OPTIMIZATION ({len(depths)*len(lrs)*len(regs)} combinations) ===")

for d in depths:
    for lr in lrs:
        for r in regs:
            print(f"Training: depth={d}, lr={lr}, reg={r}...", end=" ", flush=True)
            model = CatBoostClassifier(
                iterations=500, # Reduced iterations for speed
                depth=d,
                learning_rate=lr,
                l2_leaf_reg=r,
                loss_function='Logloss',
                eval_metric='AUC',
                random_seed=42,
                verbose=False,
                cat_features=final_cat_features,
                early_stopping_rounds=50,
                auto_class_weights='Balanced'
            )
            model.fit(X_train_final, y_train, eval_set=(X_test_final, y_test))
            
            y_prob = model.predict_proba(X_test_final)[:, 1]
            y_pred = model.predict(X_test_final)
            
            auc = roc_auc_score(y_test, y_prob)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            
            print(f"Done. AUC: {auc:.4f}")
            
            results.append({
                'depth': d, 'lr': lr, 'reg': r,
                'auc': auc, 'precision': prec, 'recall': rec,
                'model': model
            })

results_df = pd.DataFrame(results).sort_values(by='auc', ascending=False)
print(f"\n=== CATBOOST AUC OPTIMIZATION RESULTS ===")
print(results_df[['depth', 'lr', 'reg', 'auc', 'precision', 'recall']].to_string(index=False))

best_result = results_df.iloc[0]
best_auc = best_result['auc']
best_recall = best_result['recall']

# Model Selection Rule
if best_auc > baseline_auc and best_recall >= 0.95:
    final_model = best_result['model']
    choice = "CatBoost"
    justification = f"CatBoost (AUC: {best_auc:.4f}) outperformed Baseline (AUC: {baseline_auc:.4f}) and met Recall target ({best_recall:.4f} >= 0.95)."
    joblib.dump(final_model, FINAL_MODEL_PATH)
else:
    final_model = lr_pipeline
    choice = "Logistic Regression"
    justification = f"CatBoost did not meet both AUC improvement and Recall >= 0.95. Retaining Logistic Regression."
    joblib.dump(final_model, FINAL_MODEL_PATH)

print(f"\n=== FINAL MODEL SUMMARY ===")
print(f"Best CatBoost Hyperparameters: depth={best_result['depth']}, lr={best_result['lr']}, reg={best_result['reg']}")
print(f"Best CatBoost ROC-AUC: {best_auc:.4f}")
print(f"Baseline ROC-AUC: {baseline_auc:.4f}")
print(f"Final Model Choice: {choice}")
print(f"Justification: {justification}")
print(f"Model saved successfully as {FINAL_MODEL_PATH}.")
