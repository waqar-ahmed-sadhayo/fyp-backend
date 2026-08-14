import io

from app.ml.predictor import DISEASES

from conftest import register


def test_list_diseases_is_public(client):
    res = client.get("/api/diseases")
    assert res.status_code == 200
    body = res.get_json()
    for d in DISEASES:
        assert d in body
        assert "features" in body[d]
        assert "metrics" in body[d]


def test_diseases_info_is_public(client):
    res = client.get("/api/diseases/info")
    assert res.status_code == 200
    body = res.get_json()
    for d in DISEASES:
        assert d in body
        assert "overview" in body[d]
        assert "symptoms" in body[d]
        assert "risk_factors" in body[d]
        assert "prevention" in body[d]


def test_predict_requires_auth(client):
    res = client.post("/api/predict/heart", json={})
    assert res.status_code == 401


def test_predict_unknown_disease(client, auth_headers):
    res = client.post("/api/predict/not-a-disease", json={}, headers=auth_headers)
    assert res.status_code == 404


def test_predict_each_disease_with_empty_payload(client, auth_headers):
    # missing/empty feature values fall back to median imputation in predictor.predict(),
    # so an empty payload should still produce a valid prediction for every model.
    for disease in DISEASES:
        res = client.post(f"/api/predict/{disease}", json={}, headers=auth_headers)
        assert res.status_code == 201, (disease, res.get_json())
        body = res.get_json()
        assert body["disease"] == disease
        assert body["prediction"]
        assert 0.0 <= body["probability"] <= 1.0
        assert body["model_used"]


def test_predict_includes_shap_explanation(client, auth_headers):
    res = client.post("/api/predict/heart", json={}, headers=auth_headers)
    assert res.status_code == 201
    explanation = res.get_json()["explanation"]
    assert explanation is not None
    assert 1 <= len(explanation) <= 5
    for item in explanation:
        assert set(item.keys()) == {"feature", "value", "contribution"}
        assert isinstance(item["contribution"], (int, float))
    # ranked by |contribution|, most influential feature first
    contributions = [abs(item["contribution"]) for item in explanation]
    assert contributions == sorted(contributions, reverse=True)


def test_predict_explanation_matches_chosen_model_type(client, auth_headers):
    # LR/RF/GB/HistGB-chosen diseases get a fast exact explanation; an
    # SVM-chosen disease deliberately gets None instead of a slow approximate
    # one — see app/ml/explain.py's docstring. Whichever model won this
    # training run, the explanation field should be consistent with it.
    for disease in DISEASES:
        res = client.post(f"/api/predict/{disease}", json={}, headers=auth_headers)
        assert res.status_code == 201, (disease, res.get_json())
        body = res.get_json()
        if body["model_used"] == "svm":
            assert body["explanation"] is None, disease
        else:
            assert body["explanation"] is not None, disease


def test_predict_saves_to_history(client, auth_headers):
    res = client.post("/api/predict/heart", json={}, headers=auth_headers)
    result_id = res.get_json()["id"]

    hist = client.get("/api/history", headers=auth_headers)
    assert hist.status_code == 200
    ids = [r["id"] for r in hist.get_json()]
    assert result_id in ids


def test_history_filter_by_disease(client, auth_headers):
    client.post("/api/predict/heart", json={}, headers=auth_headers)
    client.post("/api/predict/diabetes", json={}, headers=auth_headers)

    res = client.get("/api/history?disease=heart", headers=auth_headers)
    body = res.get_json()
    assert len(body) == 1
    assert body[0]["disease"] == "heart"


def test_history_requires_auth(client):
    res = client.get("/api/history")
    assert res.status_code == 401


def test_history_is_scoped_to_the_user(client, auth_headers):
    client.post("/api/predict/heart", json={}, headers=auth_headers)

    other = register(client, email="other@example.com")
    other_headers = {"Authorization": f"Bearer {other['token']}"}

    res = client.get("/api/history", headers=other_headers)
    assert res.get_json() == []


def test_delete_history(client, auth_headers):
    created = client.post("/api/predict/heart", json={}, headers=auth_headers)
    result_id = created.get_json()["id"]

    res = client.delete(f"/api/history/{result_id}", headers=auth_headers)
    assert res.status_code == 200

    hist = client.get("/api/history", headers=auth_headers)
    assert hist.get_json() == []


def test_delete_history_not_owned_by_user(client, auth_headers):
    created = client.post("/api/predict/heart", json={}, headers=auth_headers)
    result_id = created.get_json()["id"]

    other = register(client, email="other2@example.com")
    other_headers = {"Authorization": f"Bearer {other['token']}"}

    res = client.delete(f"/api/history/{result_id}", headers=other_headers)
    assert res.status_code == 404


def test_predict_upload_csv(client, auth_headers):
    csv_bytes = io.BytesIO(b"age,trestbps\n52,130\n")
    res = client.post(
        "/api/predict/heart/upload",
        data={"file": (csv_bytes, "panel.csv")},
        headers=auth_headers,
        content_type="multipart/form-data",
    )
    assert res.status_code == 201
    assert res.get_json()["disease"] == "heart"


def test_predict_upload_rejects_non_csv(client, auth_headers):
    bad_file = io.BytesIO(b"not a csv")
    res = client.post(
        "/api/predict/heart/upload",
        data={"file": (bad_file, "panel.txt")},
        headers=auth_headers,
        content_type="multipart/form-data",
    )
    assert res.status_code == 400


def test_predict_upload_rejects_empty_csv(client, auth_headers):
    empty_csv = io.BytesIO(b"age,trestbps\n")
    res = client.post(
        "/api/predict/heart/upload",
        data={"file": (empty_csv, "panel.csv")},
        headers=auth_headers,
        content_type="multipart/form-data",
    )
    assert res.status_code == 400


def test_predict_upload_rejects_binary_disguised_as_csv(client, auth_headers):
    fake_csv = io.BytesIO(b"\x00\x01\xfe\xff not really a csv")
    res = client.post(
        "/api/predict/heart/upload",
        data={"file": (fake_csv, "panel.csv")},
        headers=auth_headers,
        content_type="multipart/form-data",
    )
    assert res.status_code == 400


def test_predict_upload_rejects_unrecognized_columns(client, auth_headers):
    unrelated_csv = io.BytesIO(b"favorite_color,shoe_size\nblue,10\n")
    res = client.post(
        "/api/predict/heart/upload",
        data={"file": (unrelated_csv, "panel.csv")},
        headers=auth_headers,
        content_type="multipart/form-data",
    )
    assert res.status_code == 400


# ---------------------------------------------------------------------
# CKD/liver diagnosis fixups: malformed-input rejection, and the sample
# healthy/diseased/borderline panels from the bug report this was written
# against (see PR description / conversation for the original symptom).
# ---------------------------------------------------------------------

CKD_NEGATIVE_SAMPLE = {
    "age": 34, "bp": 80, "sg": 1.020, "al": 0, "su": 0, "rbc": 1, "pc": 1,
    "pcc": 0, "ba": 0, "bgr": 98, "bu": 32, "sc": 0.9, "sod": 141, "pot": 4.2,
    "hemo": 15.4, "pcv": 46, "wc": 7800, "rc": 5.2, "htn": 0, "dm": 0,
    "cad": 0, "appet": 0, "pe": 0, "ane": 0,  # appet=0 -> "Good"
}

CKD_POSITIVE_SAMPLE = {
    "age": 62, "bp": 90, "sg": 1.010, "al": 3, "su": 0, "rbc": 0, "pc": 0,
    "pcc": 1, "ba": 1, "bgr": 157, "bu": 76, "sc": 4.1, "sod": 129, "pot": 5.2,
    "hemo": 9.2, "pcv": 29, "wc": 12700, "rc": 3.4, "htn": 1, "dm": 1,
    "cad": 0, "appet": 1, "pe": 1, "ane": 1,  # appet=1 -> "Poor"
}

LIVER_NEGATIVE_SAMPLE = {
    "Age": 29, "Gender": 0, "Total_Bilirubin": 0.7, "Direct_Bilirubin": 0.2,
    "Alkaline_Phosphotase": 187, "Alamine_Aminotransferase": 22,
    "Aspartate_Aminotransferase": 24, "Total_Protiens": 7.0, "Albumin": 4.2,
    "Albumin_and_Globulin_Ratio": 1.5,
}


def test_ckd_healthy_panel_predicts_negative(client, auth_headers):
    res = client.post("/api/predict/kidney", json=CKD_NEGATIVE_SAMPLE, headers=auth_headers)
    assert res.status_code == 201, res.get_json()
    assert res.get_json()["prediction"] == "negative"


def test_ckd_diseased_panel_predicts_positive(client, auth_headers):
    res = client.post("/api/predict/kidney", json=CKD_POSITIVE_SAMPLE, headers=auth_headers)
    assert res.status_code == 201, res.get_json()
    assert res.get_json()["prediction"] == "positive"


def test_liver_healthy_panel_does_not_predict_positive(client, auth_headers):
    res = client.post("/api/predict/liver", json=LIVER_NEGATIVE_SAMPLE, headers=auth_headers)
    assert res.status_code == 201, res.get_json()
    assert res.get_json()["prediction"] in ("negative", "borderline")


def test_liver_zero_bilirubin_does_not_crash_on_divide_by_zero(client, auth_headers):
    sample = dict(LIVER_NEGATIVE_SAMPLE, Total_Bilirubin=0, Alamine_Aminotransferase=0)
    res = client.post("/api/predict/liver", json=sample, headers=auth_headers)
    assert res.status_code == 201, res.get_json()
    assert res.get_json()["prediction"]


def test_predict_rejects_non_numeric_field(client, auth_headers):
    res = client.post(
        "/api/predict/kidney",
        json={**CKD_NEGATIVE_SAMPLE, "sc": "not-a-number"},
        headers=auth_headers,
    )
    assert res.status_code == 400
    body = res.get_json()
    assert body["error"] == "invalid input"
    assert any("sc" in f for f in body["fields"])


def test_predict_rejects_out_of_range_categorical(client, auth_headers):
    res = client.post(
        "/api/predict/kidney",
        json={**CKD_NEGATIVE_SAMPLE, "appet": 2},
        headers=auth_headers,
    )
    assert res.status_code == 400
    body = res.get_json()
    assert any("appet" in f for f in body["fields"])


def test_predict_does_not_silently_impute_a_malformed_field(client, auth_headers):
    # A present-but-garbage value must be rejected, not silently coerced to
    # NaN and imputed the way `float(v)` used to swallow it.
    res = client.post(
        "/api/predict/heart",
        json={"age": "fifty-two"},
        headers=auth_headers,
    )
    assert res.status_code == 400
