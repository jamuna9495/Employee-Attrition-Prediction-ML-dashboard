"""
logic.py
--------
Pure-Python (no Streamlit) helpers: data loading, model loading/training,
and single/batch prediction. Kept separate from app.py so it can be
unit-tested without a Streamlit runtime.
"""

import os
import pandas as pd
import numpy as np
import joblib

MODEL_PATH = "model_artifacts.pkl"

WINDOWS_DEFAULT_PATH = r"E:\DS and ML\HR analysis (2).csv"
LOCAL_DEFAULT_PATH = "HR_data.csv"

DROP_COLS = [
    "CF_current Employee", "CF_attrition label", "CF_age band",
    "Employee Count", "Standard Hours",
]

CATEGORICAL_COLS = [
    "Business Travel", "Department", "Education Field", "Gender",
    "Job Role", "Marital Status", "Over Time", "Education",
]

TARGET = "Attrition"


def resolve_data_path():
    """Try the user's Windows path first, then a local copy next to the app."""
    if os.path.exists(WINDOWS_DEFAULT_PATH):
        return WINDOWS_DEFAULT_PATH
    if os.path.exists(LOCAL_DEFAULT_PATH):
        return LOCAL_DEFAULT_PATH
    return None


def load_raw_data(path=None):
    path = path or resolve_data_path()
    if path is None:
        return None
    df = pd.read_csv(path)
    return df


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def train_and_save_model(df):
    """Trains a fresh model from a raw dataframe and saves artifacts. Returns artifacts dict."""
    import train_model as tm
    clean = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    X, y, encoders, feature_names = tm.build_dataset(clean)
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    importances = dict(sorted(
        zip(feature_names, model.feature_importances_.tolist()),
        key=lambda kv: kv[1], reverse=True
    ))
    artifacts = {
        "model": model, "model_name": "GradientBoosting", "encoders": encoders,
        "feature_names": feature_names, "categorical_cols": CATEGORICAL_COLS,
        "feature_importances": importances,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "roc_auc": roc_auc_score(y_test, y_proba),
        "class_balance": y.value_counts().to_dict(),
    }
    joblib.dump(artifacts, MODEL_PATH)
    return artifacts


def prepare_features(input_dict, artifacts):
    """Turn a single employee's raw field dict into a model-ready row (1-row DataFrame)."""
    row = {}
    encoders = artifacts["encoders"]
    for feat in artifacts["feature_names"]:
        val = input_dict.get(feat)
        if feat in CATEGORICAL_COLS:
            le = encoders[feat]
            if val not in le.classes_:
                val = le.classes_[0]
            row[feat] = le.transform([val])[0]
        else:
            row[feat] = val
    return pd.DataFrame([row])[artifacts["feature_names"]]


def predict_single(input_dict, artifacts, threshold=0.5):
    X = prepare_features(input_dict, artifacts)
    model = artifacts["model"]
    proba = model.predict_proba(X)[0, 1]
    pred = int(proba >= threshold)
    return pred, proba


def risk_bucket(proba):
    if proba >= 0.6:
        return "High Risk", "#e63946"
    elif proba >= 0.3:
        return "Medium Risk", "#f4a261"
    else:
        return "Low Risk", "#2a9d8f"


def score_bulk(df, artifacts):
    """Vectorized scoring for an entire dataframe. Returns a numpy array of probabilities."""
    clean = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore").copy()
    for feat in artifacts["feature_names"]:
        if feat in CATEGORICAL_COLS:
            le = artifacts["encoders"][feat]
            valid = set(le.classes_)
            clean[feat] = clean[feat].astype(str).apply(lambda v: v if v in valid else le.classes_[0])
            clean[feat] = le.transform(clean[feat])
    X = clean[artifacts["feature_names"]]
    return artifacts["model"].predict_proba(X)[:, 1]


def shap_explain(input_dict, artifacts):
    """
    Try to produce true SHAP values for a single prediction (tree models only).
    Returns a list of (feature, shap_value, feature_value) sorted by |impact|, or
    None if shap isn't installed / model isn't tree-based.
    """
    try:
        import shap
    except ImportError:
        return None

    model = artifacts["model"]
    if not hasattr(model, "estimators_") and not hasattr(model, "tree_"):
        return None

    try:
        X = prepare_features(input_dict, artifacts)
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X)
        # GradientBoosting / RandomForest binary classifiers can return different shapes
        if isinstance(raw, list):
            values = raw[1][0] if len(raw) > 1 else raw[0][0]
        else:
            values = raw[0]
            if hasattr(values, "ndim") and values.ndim > 1:
                values = values[:, 1]
        pairs = list(zip(artifacts["feature_names"], values, X.iloc[0].tolist()))
        pairs.sort(key=lambda t: abs(t[1]), reverse=True)
        return pairs
    except Exception:
        return None


def what_if_curve(input_dict, artifacts, feature, value_range):
    """Sweep one feature across value_range, holding everything else fixed, return probabilities."""
    probs = []
    for v in value_range:
        trial = dict(input_dict)
        trial[feature] = v
        _, p = predict_single(trial, artifacts)
        probs.append(p)
    return probs


def top_contributing_factors(input_dict, artifacts, df_reference, n=4):
    """
    Lightweight explainability: rank the employee's features by
    (global feature importance) x (how far their value deviates from the
    average value of employees who left), for numeric features only.
    """
    importances = artifacts["feature_importances"]
    numeric_feats = [f for f in artifacts["feature_names"] if f not in CATEGORICAL_COLS]

    leavers = df_reference[df_reference[TARGET] == "Yes"]
    stayers = df_reference[df_reference[TARGET] == "No"]

    scored = []
    for feat in numeric_feats:
        imp = importances.get(feat, 0)
        if imp <= 0 or feat not in input_dict:
            continue
        leaver_mean = leavers[feat].mean()
        stayer_mean = stayers[feat].mean()
        spread = df_reference[feat].std() or 1
        direction_toward_leavers = abs(input_dict[feat] - stayer_mean) / spread
        scored.append((feat, imp * direction_toward_leavers, input_dict[feat], leaver_mean, stayer_mean))

    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:n]
