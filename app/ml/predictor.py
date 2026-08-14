"""Loads trained model bundles once and exposes predict()/build_model_input() helpers."""
import logging
import os

import joblib
import numpy as np
import pandas as pd

from .feature_engineering import add_derived_features

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

DISEASES = ["heart", "diabetes", "breast_cancer", "kidney", "liver"]

# Every training script encodes the target the same way: 1 = disease
# present. Confirmed against each train_<disease>() function — never assume
# this, always verify it against the actual label construction when adding
# a new disease.
POSITIVE_CLASS_INDEX = 1

# A predicted probability within this many points of the disease's own
# tuned decision threshold is reported as "borderline" rather than a
# confident positive/negative — a near-coin-flip score presented as a firm
# label is misleading. Measured against each disease's actual threshold
# (not a hardcoded 0.5), since thresholds are legitimately tuned per model
# (see train_models.py::best_threshold_for_f1) and aren't all 0.5.
BORDERLINE_MARGIN = 0.05

_REQUIRED_BUNDLE_KEYS = (
    "model", "imputer", "scaler", "feature_names", "model_feature_names",
    "metrics", "label_map",
)

logger = logging.getLogger("mdds.predict")

_bundles = {}


def _load(disease: str):
    if disease not in _bundles:
        path = os.path.join(MODEL_DIR, f"{disease}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No trained model for '{disease}'. Run train_models.py first."
            )
        try:
            bundle = joblib.load(path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model bundle for '{disease}' from {path}: {e}"
            ) from e

        missing = [k for k in _REQUIRED_BUNDLE_KEYS if k not in bundle]
        if missing:
            raise RuntimeError(
                f"Model bundle for '{disease}' is missing required key(s) {missing} "
                f"— it looks like it was saved by an older/incompatible version of "
                f"train_models.py. Retrain it before serving predictions."
            )
        _bundles[disease] = bundle
    return _bundles[disease]


def load_all():
    """Eagerly loads and validates every disease's model bundle. Call this
    once at app startup (see app/__init__.py) so a missing/corrupt/
    incompatible bundle fails loudly at boot, not silently on some user's
    first prediction request."""
    for d in DISEASES:
        _load(d)


def get_feature_names(disease: str):
    return _load(disease)["feature_names"]


def get_metrics(disease: str):
    return _load(disease)["metrics"]


def get_bundle(disease: str):
    return _load(disease)


def build_model_input(disease: str, payload: dict):
    """Builds the model-ready (imputed + scaled) feature row for one raw
    payload, applying the same derived features used at training time
    (feature_engineering.add_derived_features). Shared by predict() and
    explain.py so they can never see a different feature set than the model
    was actually trained on.

    Returns (X_scaled, model_feature_names) — the latter for labeling SHAP
    contributions against the right (derived-inclusive) column names.
    """
    bundle = _load(disease)
    raw_features = bundle["feature_names"]
    model_features = bundle["model_feature_names"]

    raw_row = {}
    for f in raw_features:
        v = payload.get(f, None)
        try:
            raw_row[f] = float(v) if v is not None and v != "" else np.nan
        except (TypeError, ValueError):
            raw_row[f] = np.nan

    df = pd.DataFrame([raw_row])
    df = add_derived_features(df, disease)
    df = df.reindex(columns=model_features)

    X = df.to_numpy(dtype=float)
    X_imp = bundle["imputer"].transform(X)
    X_scaled = bundle["scaler"].transform(X_imp)
    return X_scaled, model_features


def predict(disease: str, payload: dict):
    """
    payload: dict of feature_name -> value (missing keys become NaN / imputed)
    returns: (label:str, probability:float, model_name:str)

    `label` is one of the disease's label_map values (e.g. "positive"/
    "negative", or "benign"/"malignant"), or the generic "borderline" when
    the positive-class probability lands within BORDERLINE_MARGIN of the
    disease's own decision threshold — a near-coin-flip score that
    shouldn't be presented with the same visual confidence as a clear call.
    """
    bundle = _load(disease)
    X_scaled, _ = build_model_input(disease, payload)
    label_map = bundle["label_map"]
    threshold = bundle["metrics"].get("decision_threshold", 0.5)
    model = bundle["model"]
    model_name = bundle["metrics"]["chosen_model"]

    pos_proba = None
    if hasattr(model, "predict_proba"):
        proba_vec = model.predict_proba(X_scaled)[0]
        pos_proba = float(proba_vec[POSITIVE_CLASS_INDEX])
        pred_class = int(pos_proba >= threshold)
        proba = float(proba_vec[pred_class])
        if abs(pos_proba - threshold) <= BORDERLINE_MARGIN:
            label = "borderline"
        else:
            label = label_map.get(pred_class, str(pred_class))
    else:
        pred_class = int(model.predict(X_scaled)[0])
        proba = None
        label = label_map.get(pred_class, str(pred_class))

    # Server-side only, no PII (no name/email — just the disease, the
    # already-scaled/anonymous feature vector, and the decision inputs) so
    # any future "this looks wrong" report can be traced end-to-end.
    logger.info(
        "predict disease=%s model=%s threshold=%.4f pos_proba=%s label=%s scaled_features=%s",
        disease, model_name, threshold,
        f"{pos_proba:.4f}" if pos_proba is not None else "n/a",
        label, np.round(X_scaled[0], 4).tolist(),
    )

    return label, round(proba, 4) if proba is not None else None, model_name
