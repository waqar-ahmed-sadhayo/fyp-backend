"""Input validation for prediction payloads.

Policy — deliberately asymmetric between "missing" and "malformed":

  - A field that is absent, None, or an empty string is MISSING. It is left
    as NaN and median-imputed downstream (predictor.build_model_input).
    This is intentional, not an oversight: the CSV/PDF upload endpoints are
    explicitly designed to work from a partial lab panel (see predict.py's
    upload-pdf docstring), and this contract is pinned by
    tests/test_predict.py::test_predict_each_disease_with_empty_payload,
    which sends `{}` and expects a valid prediction back for every disease.

  - A field that IS present with a non-empty value must actually be valid.
    A value that can't be parsed as a number, or a categorical value
    outside its known encoding, is MALFORMED and is rejected outright —
    never silently coerced to NaN/0/a default the way it used to be
    (`float(v)` swallowing a ValueError into NaN).
"""
import math

# Fields whose value the trained model expects as one of a small, fixed set
# of already-encoded integers (as sent by the frontend's <select>). These
# are checked against the *actual* encoding recorded in the model bundle
# at train time (bundle["category_encodings"]) where available, falling
# back to this hardcoded set for models trained before that field existed
# or whose categoricals were never LabelEncoder'd in the first place (e.g.
# heart.csv already ships pre-encoded ints, liver's Gender is mapped by
# hand in train_liver()).
_FALLBACK_CATEGORICAL_FIELDS = {
    "heart": {
        "sex": {0, 1}, "cp": {0, 1, 2, 3}, "fbs": {0, 1}, "restecg": {0, 1, 2},
        "exang": {0, 1}, "slope": {0, 1, 2}, "thal": {1, 2, 3},
    },
    "kidney": {
        "rbc": {0, 1}, "pc": {0, 1}, "pcc": {0, 1}, "ba": {0, 1},
        "htn": {0, 1}, "dm": {0, 1}, "cad": {0, 1}, "appet": {0, 1},
        "pe": {0, 1}, "ane": {0, 1},
    },
    "liver": {"Gender": {0, 1}},
}


def _is_missing(v):
    return v is None or (isinstance(v, str) and v.strip() == "")


def _categorical_fields(disease: str, category_encodings: dict) -> dict:
    if category_encodings:
        return {f: set(mapping.values()) for f, mapping in category_encodings.items()}
    return _FALLBACK_CATEGORICAL_FIELDS.get(disease, {})


def validate_payload(disease: str, payload: dict, feature_names: list, category_encodings: dict = None) -> list:
    """Returns a list of human-readable error strings; an empty list means
    the payload is valid. Only fields actually present in `payload` are
    checked — an absent field is the deliberate "missing -> impute" case,
    not a validation error (see module docstring)."""
    errors = []
    categorical = _categorical_fields(disease, category_encodings or {})

    for f in feature_names:
        if f not in payload or _is_missing(payload[f]):
            continue

        v = payload[f]
        try:
            fv = float(v)
        except (TypeError, ValueError):
            errors.append(f"'{f}': not a number ({v!r})")
            continue

        if not math.isfinite(fv):
            errors.append(f"'{f}': must be a finite number ({v!r})")
            continue

        if f in categorical:
            allowed = categorical[f]
            if fv != int(fv) or int(fv) not in allowed:
                errors.append(f"'{f}': must be one of {sorted(allowed)} (got {v!r})")

    return errors
