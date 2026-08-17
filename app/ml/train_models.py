"""
Trains the 5 disease-detection models described in the project report:
Heart Disease, Diabetes, Breast Cancer, Kidney Disease, Liver Disease.

For each disease:
  - load + clean data
  - add domain-informed derived features (heart/diabetes/liver only — see
    feature_engineering.py)
  - impute missing values
  - scale features
  - optionally balance classes with SMOTE
  - grid-search LR / Random Forest / SVM / Gradient Boosting / Hist Gradient
    Boosting, pick the best by cross-validated F1
  - tune the decision threshold (not just the default 0.5) to maximize F1 on
    out-of-fold predictions
  - fit on full training data, evaluate on held-out test split at that threshold
  - save {model, imputer, scaler, feature_names, model_feature_names, metrics}
    as a single joblib bundle

Run: python train_models.py
Outputs land in ./models/<disease>.joblib
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (GradientBoostingClassifier,
                               HistGradientBoostingClassifier,
                               RandomForestClassifier, StackingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                      cross_val_predict, cross_val_score, train_test_split)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

from feature_engineering import add_derived_features

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODEL_DIR = os.path.join(HERE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
THRESHOLD_CANDIDATES = np.round(np.arange(0.1, 0.91, 0.01), 2)


def evaluate(model, X_test, y_test, threshold=0.5):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= threshold).astype(int)
    else:
        pred = model.predict(X_test)
        proba = pred
    return {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def best_threshold_for_f1(model, X, y, cv):
    """Sweeps decision thresholds against out-of-fold predicted probabilities
    (never the held-out test set) and returns the one maximizing F1. Classic
    "threshold moving" — often worth more on an imbalanced small dataset than
    further hyperparameter search on the default 0.5 cutoff."""
    oof_proba = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")[:, 1]
    f1s = [f1_score(y, (oof_proba >= t).astype(int), zero_division=0) for t in THRESHOLD_CANDIDATES]
    return float(THRESHOLD_CANDIDATES[int(np.argmax(f1s))])


def train_and_select(X, y, disease, use_smote=True):
    """Compare LR / RF / SVM / Gradient Boosting / Hist Gradient Boosting with
    grid-searched hyperparameters under 5-fold CV, tune the decision threshold,
    return the best fitted pipeline pieces.

    See phases/backend/phase-2-ml-pipeline-prediction-api.md for the
    before/after accuracy numbers across each round of changes to this function.
    """
    print(f"\n--- training {disease} ---", flush=True)
    raw_feature_names = list(X.columns)
    X = add_derived_features(X, disease)
    model_feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    if use_smote and y_train.value_counts(normalize=True).min() < 0.4:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)

    search_space = {
        "logistic_regression": (
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
            {"C": [0.01, 0.1, 1, 10], "class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [200, 400],
                "max_depth": [6, 10, None],
                "class_weight": [None, "balanced"],
            },
        ),
        "svm": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "class_weight": [None, "balanced"],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [100, 200],
                "max_depth": [2, 3],
                "learning_rate": [0.05, 0.1],
            },
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "max_iter": [100, 200],
                "max_depth": [3, 5, None],
                "learning_rate": [0.05, 0.1],
            },
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = {}
    stds = {}
    fitted = {}
    for name, (estimator, grid) in search_space.items():
        # n_jobs bounded (not -1): these datasets are tiny, so process-spawn
        # overhead dominates anyway, and some estimators (HistGradientBoosting
        # in particular) also parallelize internally — GridSearchCV(-1) on top
        # of that oversubscribes cores badly enough on Windows to look hung.
        search = GridSearchCV(estimator, grid, cv=cv, scoring="f1", n_jobs=2, verbose=0)
        search.fit(X_train_scaled, y_train)
        scores[name] = round(float(search.best_score_), 4)
        stds[name] = round(float(search.cv_results_["std_test_score"][search.best_index_]), 4)
        fitted[name] = search.best_estimator_
        print(f"  {name}: cv_f1={scores[name]} (+/-{stds[name]})", flush=True)

    # A stacked blend of the 3 strongest individual candidates, fed into a
    # logistic-regression meta-learner — one more candidate in the same
    # scores/stds/fitted comparison below, not a special-cased override. On
    # a well-separated dataset (breast_cancer/kidney) it simply won't beat
    # the individual winner and gets ignored; on a noisier one it sometimes
    # recovers a point or two by averaging out each base model's mistakes.
    top3 = sorted(scores, key=scores.get, reverse=True)[:3]
    stack = StackingClassifier(
        estimators=[(n, fitted[n]) for n in top3],
        final_estimator=LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        cv=cv, n_jobs=2,
    )
    stack_cv = cross_val_score(stack, X_train_scaled, y_train, cv=cv, scoring="f1")
    scores["stacking"] = round(float(stack_cv.mean()), 4)
    stds["stacking"] = round(float(stack_cv.std()), 4)
    stack.fit(X_train_scaled, y_train)
    fitted["stacking"] = stack
    print(f"  stacking({'+'.join(top3)}): cv_f1={scores['stacking']} (+/-{stds['stacking']})", flush=True)

    # One-standard-error rule (Breiman et al.) instead of a raw argmax: with
    # 5 folds on a few-hundred-row dataset, a 0.003 CV-F1 edge is well within
    # noise, not a real signal. This is exactly what picked SVM over Random
    # Forest for diabetes in an earlier round (0.8216 vs 0.8184 CV F1) and
    # then lost 4.5pp of test accuracy for it — see
    # phases/backend/phase-2-ml-pipeline-prediction-api.md. Among candidates
    # within one SE of the top score, prefer the most stable (lowest std)
    # one rather than whichever happened to land highest this split.
    best_raw = max(scores, key=scores.get)
    se = stds[best_raw] / np.sqrt(cv.get_n_splits())
    within_se = [n for n in scores if scores[n] >= scores[best_raw] - se]
    best_name = min(within_se, key=lambda n: stds[n])
    best_model = fitted[best_name]
    print(f"  -> best: {best_name} (within 1 SE of {best_raw}: {within_se}), tuning decision threshold...", flush=True)

    threshold = best_threshold_for_f1(best_model, X_train_scaled, y_train, cv)

    metrics = evaluate(best_model, X_test_scaled, y_test, threshold=threshold)
    metrics["cv_f1_scores"] = scores
    metrics["cv_f1_stds"] = stds
    metrics["chosen_model"] = best_name
    metrics["decision_threshold"] = threshold

    # A small real sample of scaled training rows, kept as the SHAP baseline
    # distribution at serve time (predictor never has access to raw training
    # data, only the saved bundle) — see explain.py.
    rng = np.random.RandomState(RANDOM_STATE)
    bg_size = min(50, X_train_scaled.shape[0])
    bg_idx = rng.choice(X_train_scaled.shape[0], size=bg_size, replace=False)
    shap_background = X_train_scaled[bg_idx]

    return best_model, imputer, scaler, metrics, raw_feature_names, model_feature_names, shap_background


def save_bundle(name, model, imputer, scaler, feature_names, model_feature_names, metrics,
                 shap_background, label_map=None, category_encodings=None):
    bundle = {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "feature_names": feature_names,              # raw inputs only — drives the frontend form
        "model_feature_names": model_feature_names,   # raw + derived — actual model input order
        "metrics": metrics,
        "shap_background": shap_background,
        "label_map": label_map or {0: "negative", 1: "positive"},
        # {column: {raw_string_value: encoded_int}} for columns that went through
        # LabelEncoder — saved so inference/validation code never has to guess or
        # reverse-engineer what a category's 0/1 actually means (see the appet
        # good/poor inversion bug this caught: the frontend had assumed the wrong
        # direction because nothing recorded the real mapping anywhere).
        "category_encodings": category_encodings or {},
    }
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    joblib.dump(bundle, path)
    print(f"\n=== {name} ===")
    print(f"chosen model: {metrics['chosen_model']}  cv_f1: {metrics['cv_f1_scores']}")
    print(f"decision threshold: {metrics['decision_threshold']}")
    print(f"test metrics: acc={metrics['accuracy']} f1={metrics['f1']} "
          f"auc={metrics['roc_auc']} precision={metrics['precision']} recall={metrics['recall']}")
    return path


# ---------------------------------------------------------------- HEART -----
def train_heart():
    df = pd.read_csv(os.path.join(DATA_DIR, "heart.csv"))
    df.columns = [c.strip().lower() for c in df.columns]
    y = (df["target"] > 0).astype(int)
    X = df.drop(columns=["target"])
    model, imputer, scaler, metrics, raw_names, model_names, bg = train_and_select(X, y, "heart")
    save_bundle("heart", model, imputer, scaler, raw_names, model_names, metrics, bg)


# -------------------------------------------------------------- DIABETES ----
def train_diabetes():
    df = pd.read_csv(os.path.join(DATA_DIR, "diabetes.csv"))
    # Zeros in these columns are missing values, not real measurements
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for c in zero_as_missing:
        df[c] = df[c].replace(0, np.nan)
    y = df["Outcome"].astype(int)
    X = df.drop(columns=["Outcome"])
    model, imputer, scaler, metrics, raw_names, model_names, bg = train_and_select(X, y, "diabetes")
    save_bundle("diabetes", model, imputer, scaler, raw_names, model_names, metrics, bg)


# ----------------------------------------------------------- BREAST CANCER --
def train_breast_cancer():
    data = load_breast_cancer(as_frame=True)
    X = data.frame.drop(columns=["target"])
    # sklearn encodes 0=malignant, 1=benign; flip so 1 = "positive for cancer" (malignant)
    y = (data.frame["target"] == 0).astype(int)
    model, imputer, scaler, metrics, raw_names, model_names, bg = train_and_select(X, y, "breast_cancer")
    save_bundle(
        "breast_cancer", model, imputer, scaler, raw_names, model_names, metrics, bg,
        label_map={0: "benign", 1: "malignant"},
    )


# --------------------------------------------------------------- KIDNEY -----
# Only these columns are genuinely categorical (yes/no, normal/abnormal,
# present/not-present). Every other column — including wc/rc/pcv, and al/su
# — is a real numeric measurement that merely *arrives* as text because the
# raw CSV uses stray "?" placeholders for missing values, which makes
# pandas infer an `object` dtype for the whole column.
#
# This used to be detected with `select_dtypes(include="object")`, which
# swept wc/rc/pcv into the same LabelEncoder loop as the true categoricals.
# LabelEncoder replaces each unique string with an arbitrary alphabetical
# rank (e.g. wc="7800" became rank 69 of 89), destroying the real magnitude.
# Inference-time code has no way to reproduce that rank — it just casts the
# raw value to float — so a real wc of 7800 got fed into a scaler fit on
# ranks 0-88, landing hundreds of standard deviations out of distribution
# and saturating every prediction toward "positive" regardless of the rest
# of the panel. That was the dominant cause of healthy-looking CKD inputs
# scoring as strongly positive.
KIDNEY_CATEGORICAL_COLS = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]


def train_kidney():
    df = pd.read_csv(os.path.join(DATA_DIR, "kidney.csv"))
    df = df.drop(columns=["id"], errors="ignore")
    df.columns = [c.strip() for c in df.columns]

    # Clean stray whitespace / "?" placeholders across every text-ish column
    # (this includes numeric columns like wc/rc/pcv, which is exactly why
    # they can't be told apart from true categoricals by dtype alone).
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip().replace(
            {"nan": np.nan, "?": np.nan, "\tno": "no", "\tyes": "yes", " yes": "yes"}
        )

    df["classification"] = df["classification"].str.strip().replace(
        {"ckd\t": "ckd", "notckd": "notckd"}
    )
    y = (df["classification"] == "ckd").astype(int)
    X = df.drop(columns=["classification"])

    # Label-encode ONLY the true categorical columns, and record the exact
    # mapping used (see save_bundle's category_encodings) so serving-side
    # validation can check incoming values against reality instead of an
    # assumed alphabetical order.
    category_encodings = {}
    for c in KIDNEY_CATEGORICAL_COLS:
        le = LabelEncoder()
        known = X[c].dropna().unique()
        le.fit(known)
        category_encodings[c] = {cls: int(code) for code, cls in enumerate(le.classes_)}
        X[c] = X[c].map(lambda v: le.transform([v])[0] if pd.notna(v) and v in known else np.nan)

    # Everything else (age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo,
    # pcv, wc, rc, ...) is a real numeric measurement — coerce straight to
    # numeric, never through LabelEncoder.
    X = X.apply(pd.to_numeric, errors="coerce")

    model, imputer, scaler, metrics, raw_names, model_names, bg = train_and_select(X, y, "kidney")
    save_bundle("kidney", model, imputer, scaler, raw_names, model_names, metrics, bg,
                category_encodings=category_encodings)


# ---------------------------------------------------------------- LIVER -----
def train_liver():
    cols = [
        "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
        "Alkaline_Phosphotase", "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase", "Total_Protiens", "Albumin",
        "Albumin_and_Globulin_Ratio", "Dataset",
    ]
    df = pd.read_csv(os.path.join(DATA_DIR, "liver.csv"), header=None, names=cols)
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    # Dataset: 1 = liver disease, 2 = no disease -> convert to 1/0
    y = (df["Dataset"] == 1).astype(int)
    X = df.drop(columns=["Dataset"])
    model, imputer, scaler, metrics, raw_names, model_names, bg = train_and_select(X, y, "liver")
    save_bundle("liver", model, imputer, scaler, raw_names, model_names, metrics, bg)


if __name__ == "__main__":
    train_heart()
    train_diabetes()
    train_breast_cancer()
    train_kidney()
    train_liver()

    # Write a combined metrics summary for the frontend / report
    summary = {}
    for name in ["heart", "diabetes", "breast_cancer", "kidney", "liver"]:
        bundle = joblib.load(os.path.join(MODEL_DIR, f"{name}.joblib"))
        summary[name] = {
            "chosen_model": bundle["metrics"]["chosen_model"],
            "accuracy": bundle["metrics"]["accuracy"],
            "f1": bundle["metrics"]["f1"],
            "roc_auc": bundle["metrics"]["roc_auc"],
            "precision": bundle["metrics"]["precision"],
            "recall": bundle["metrics"]["recall"],
            "decision_threshold": bundle["metrics"]["decision_threshold"],
            "feature_names": bundle["feature_names"],
        }
    with open(os.path.join(MODEL_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nAll models trained. Summary written to models/summary.json")
