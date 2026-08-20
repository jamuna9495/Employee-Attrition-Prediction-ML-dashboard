"""
train_model.py
----------------
Trains the Employee Attrition prediction model used by the Streamlit app.
Run this once (or whenever the CSV changes) to (re)generate model_artifacts.pkl

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

DATA_PATH = "HR_data.csv"          # local copy used for training in this environment
OUTPUT_PATH = "model_artifacts.pkl"

# Columns that leak the target or carry no signal (constant / duplicate columns)
DROP_COLS = [
    "CF_current Employee",   # 1:1 derived from Attrition -> leakage
    "CF_attrition label",    # duplicate of Attrition -> leakage
    "CF_age band",           # redundant with Age
    "Employee Count",        # constant
    "Standard Hours",        # constant
]

CATEGORICAL_COLS = [
    "Business Travel", "Department", "Education Field", "Gender",
    "Job Role", "Marital Status", "Over Time", "Education",
]

TARGET = "Attrition"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    return df


def build_dataset(df):
    df = df.copy()
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET])

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    feature_names = list(X.columns)
    return X, y, encoders, feature_names


def main():
    print("Loading data...")
    df = load_data()
    X, y, encoders, feature_names = build_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
    }

    results = {}
    best_name, best_model, best_auc = None, None, -1

    print("\nTraining & evaluating candidate models (5-fold CV on ROC-AUC)...")
    for name, model in candidates.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        results[name] = metrics
        print(f"  {name:20s} | CV-AUC {metrics['cv_auc_mean']:.3f} | Test-AUC {metrics['roc_auc']:.3f} "
              f"| F1 {metrics['f1']:.3f} | Recall {metrics['recall']:.3f}")

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_name = name
            best_model = model

    print(f"\nBest model: {best_name} (Test ROC-AUC = {best_auc:.3f})")

    # Feature importance (for tree models) or coefficients (for logistic regression)
    if hasattr(best_model, "feature_importances_"):
        importances = dict(zip(feature_names, best_model.feature_importances_.tolist()))
    else:
        importances = dict(zip(feature_names, np.abs(best_model.coef_[0]).tolist()))
    importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))

    fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])

    artifacts = {
        "model": best_model,
        "model_name": best_name,
        "encoders": encoders,
        "feature_names": feature_names,
        "categorical_cols": CATEGORICAL_COLS,
        "results": results,
        "feature_importances": importances,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "test_shape": X_test.shape,
        "train_shape": X_train.shape,
        "class_balance": y.value_counts().to_dict(),
        "raw_columns": list(df.columns),
    }

    joblib.dump(artifacts, OUTPUT_PATH)
    print(f"\nSaved trained artifacts -> {OUTPUT_PATH}")
    print("\nClassification report (best model):")
    print(classification_report(y_test, best_model.predict(X_test), target_names=["No", "Yes"]))


if __name__ == "__main__":
    main()
