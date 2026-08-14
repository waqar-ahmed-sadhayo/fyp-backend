"""Regression coverage for two real bugs found in the kidney (CKD) pipeline:

1. wc/rc/pcv (real numeric lab values that arrive as text in the raw CSV
   because of stray "?" placeholders) were being routed through
   LabelEncoder alongside genuine categoricals, replacing their real
   magnitude with an arbitrary alphabetical rank — see train_models.py's
   KIDNEY_CATEGORICAL_COLS comment for the full story.

2. The frontend's `appet` Good/Poor option values were inverted relative
   to what the model actually learned (good=0, poor=1), silently telling
   the model the opposite of what the user selected.

These tests pin the fix so either regression is caught immediately instead
of only surfacing as "healthy-looking inputs score positive" in production.
"""
from app.ml.predictor import get_bundle


def test_kidney_wc_rc_pcv_are_not_label_encoded():
    """A label-encoded rank tops out at a few dozen; a real WBC count is in
    the thousands. If this scaler mean/scale ever looks like a rank again
    (single/double digits), the LabelEncoder bug is back."""
    bundle = get_bundle("kidney")
    names = bundle["model_feature_names"]
    mean = bundle["scaler"].mean_

    wc_mean = mean[names.index("wc")]
    rc_mean = mean[names.index("rc")]
    pcv_mean = mean[names.index("pcv")]

    assert wc_mean > 1000, f"wc scaler mean is {wc_mean:.1f} — looks label-encoded, not a real WBC count"
    assert 1 < rc_mean < 10, f"rc scaler mean is {rc_mean:.1f} — expected a real RBC count (millions/cmm)"
    assert 10 < pcv_mean < 60, f"pcv scaler mean is {pcv_mean:.1f} — expected a real packed-cell-volume %"


def test_kidney_category_encodings_match_expected_direction():
    """Every categorical field's 0/1 meaning, pinned against the actual
    LabelEncoder mapping recorded at train time. If a retrain (e.g. on an
    updated dataset) ever flips one of these, this fails loudly instead of
    silently shipping an inverted field like `appet` did."""
    bundle = get_bundle("kidney")
    expected = {
        "rbc": {"abnormal": 0, "normal": 1},
        "pc": {"abnormal": 0, "normal": 1},
        "pcc": {"notpresent": 0, "present": 1},
        "ba": {"notpresent": 0, "present": 1},
        "htn": {"no": 0, "yes": 1},
        "dm": {"no": 0, "yes": 1},
        "cad": {"no": 0, "yes": 1},
        "appet": {"good": 0, "poor": 1},
        "pe": {"no": 0, "yes": 1},
        "ane": {"no": 0, "yes": 1},
    }
    assert bundle["category_encodings"] == expected
