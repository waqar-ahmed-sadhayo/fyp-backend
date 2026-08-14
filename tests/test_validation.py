from app.ml.validation import validate_payload

KIDNEY_FEATURES = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu",
    "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn", "dm", "cad",
    "appet", "pe", "ane",
]
KIDNEY_ENCODINGS = {
    "rbc": {"abnormal": 0, "normal": 1}, "pc": {"abnormal": 0, "normal": 1},
    "pcc": {"notpresent": 0, "present": 1}, "ba": {"notpresent": 0, "present": 1},
    "htn": {"no": 0, "yes": 1}, "dm": {"no": 0, "yes": 1}, "cad": {"no": 0, "yes": 1},
    "appet": {"good": 0, "poor": 1}, "pe": {"no": 0, "yes": 1}, "ane": {"no": 0, "yes": 1},
}


def test_empty_payload_is_valid_every_field_missing():
    # Pins the documented "missing -> median-impute" contract — an absent
    # field must never be flagged as a validation error.
    assert validate_payload("kidney", {}, KIDNEY_FEATURES, KIDNEY_ENCODINGS) == []


def test_none_and_empty_string_values_are_treated_as_missing():
    payload = {"age": None, "bp": "", "sc": "0.9"}
    assert validate_payload("kidney", payload, KIDNEY_FEATURES, KIDNEY_ENCODINGS) == []


def test_non_numeric_value_is_rejected():
    errors = validate_payload("kidney", {"age": "banana"}, KIDNEY_FEATURES, KIDNEY_ENCODINGS)
    assert len(errors) == 1
    assert "age" in errors[0]


def test_infinite_value_is_rejected():
    errors = validate_payload("kidney", {"sc": "inf"}, KIDNEY_FEATURES, KIDNEY_ENCODINGS)
    assert len(errors) == 1
    assert "sc" in errors[0]


def test_categorical_value_outside_known_encoding_is_rejected():
    errors = validate_payload("kidney", {"appet": 2}, KIDNEY_FEATURES, KIDNEY_ENCODINGS)
    assert len(errors) == 1
    assert "appet" in errors[0]


def test_categorical_non_integer_value_is_rejected():
    errors = validate_payload("kidney", {"htn": 0.5}, KIDNEY_FEATURES, KIDNEY_ENCODINGS)
    assert len(errors) == 1
    assert "htn" in errors[0]


def test_valid_categorical_and_numeric_values_pass():
    payload = {"age": 34, "appet": 0, "htn": 1, "sc": 0.9}
    assert validate_payload("kidney", payload, KIDNEY_FEATURES, KIDNEY_ENCODINGS) == []


def test_multiple_malformed_fields_all_reported():
    payload = {"age": "old", "appet": 9}
    errors = validate_payload("kidney", payload, KIDNEY_FEATURES, KIDNEY_ENCODINGS)
    assert len(errors) == 2
