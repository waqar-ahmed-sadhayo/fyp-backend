"""Per-prediction SHAP explanations.

Dispatches to a fast, exact SHAP algorithm per chosen model type rather than
a single "universal" explainer — that was the first version of this module,
and it measured ~17 seconds per explanation via PermutationExplainer, which
is unusable for a live API request. TreeExplainer (RF/GB/HistGB) and
LinearExplainer (LR) are both near-instant and exact.

There's no equivalently fast *exact* algorithm for a kernel SVM, so
`explain()` returns `None` for SVM-chosen models rather than paying a
multi-second latency tax for an approximate answer — see
phases/backend/phase-2-ml-pipeline-prediction-api.md for that tradeoff.
The same applies to a stacking-ensemble pick (see train_models.py): its
base estimators can each be explained individually, but there's no single
fast *exact* SHAP algorithm across the blended whole, so it falls into
this same "no explanation" bucket rather than an approximate one.
"""
import numpy as np

import shap

from . import predictor

TOP_N = 5

_TREE_MODELS = {"random_forest", "gradient_boosting", "hist_gradient_boosting"}

_explainers = {}


def _build_explainer(disease):
    bundle = predictor.get_bundle(disease)
    model = bundle["model"]
    model_name = bundle["metrics"]["chosen_model"]

    if model_name == "logistic_regression":
        return shap.LinearExplainer(model, bundle["shap_background"])
    if model_name in _TREE_MODELS:
        return shap.TreeExplainer(model)
    return None  # SVM — see module docstring


def _get_explainer(disease):
    if disease not in _explainers:
        _explainers[disease] = _build_explainer(disease)
    return _explainers[disease]


def _positive_class_row(raw_values):
    """Normalizes the various shapes shap's explainers return for a binary
    classifier across versions/algorithms down to one 1D per-feature array
    for the positive class."""
    if isinstance(raw_values, list):           # legacy [class0_arr, class1_arr]
        return np.asarray(raw_values[-1])[0]
    arr = np.asarray(raw_values)
    if arr.ndim == 3:                            # (n_samples, n_features, n_classes)
        return arr[0, :, -1]
    if arr.ndim == 2:                             # (n_samples, n_features)
        return arr[0]
    return arr


def explain(disease: str, payload: dict, top_n: int = TOP_N):
    """Returns the top `top_n` features (by absolute SHAP value) for this
    specific prediction, most influential first, or `None` if no fast exact
    explainer is available for the chosen model (SVM).

    `value` is the feature's standardized (scaled) value actually fed to the
    model. `contribution` is signed, in the model's margin/log-odds space
    (not a probability delta): positive pushes toward the positive/
    disease-present class, negative pushes toward negative/benign.
    """
    explainer = _get_explainer(disease)
    if explainer is None:
        return None

    X_scaled, model_feature_names = predictor.build_model_input(disease, payload)
    raw = explainer.shap_values(X_scaled) if hasattr(explainer, "shap_values") else explainer(X_scaled).values
    row = _positive_class_row(raw)

    ranked = sorted(
        zip(model_feature_names, X_scaled[0].tolist(), row.tolist()),
        key=lambda t: abs(t[2]),
        reverse=True,
    )

    return [
        {"feature": f, "value": round(v, 4), "contribution": round(c, 4)}
        for f, v, c in ranked[:top_n]
    ]
