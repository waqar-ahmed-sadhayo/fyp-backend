import csv
import io

import pdfplumber
from flask import Blueprint, jsonify, request, send_file
from flask_jwt_extended import get_jwt_identity, jwt_required

from .disease_content import DISEASE_INFO
from .ml import explain, predictor
from .ml.validation import validate_payload
from .models import TestResult, db
from .pdf_export import build_result_pdf
from .pdf_parse import extract_fields_from_pdf_text

predict_bp = Blueprint("predict", __name__, url_prefix="/api")


def _explain_safely(disease, payload):
    """SHAP explanation is a bonus on top of the prediction, not a hard
    requirement — if it fails for any reason (unexpected shap/sklearn version
    interaction, a genuinely pathological input, etc.) the prediction itself
    still succeeds and gets saved; the caller just doesn't get an `explanation`."""
    try:
        return explain.explain(disease, payload)
    except Exception:
        return None


def _validate_or_none(disease, payload):
    """Returns a (jsonify body, 400) tuple to return immediately if the
    payload has any malformed (present-but-invalid) field, else None. Never
    flags a field that's simply absent — that's the documented "missing ->
    median-impute" path, not a validation error."""
    bundle = predictor.get_bundle(disease)
    errors = validate_payload(
        disease, payload, bundle["feature_names"], bundle.get("category_encodings"),
    )
    if errors:
        return jsonify({"error": "invalid input", "fields": errors}), 400
    return None


def _save_result(disease, payload, label, probability, model_used):
    result = TestResult(
        user_id=get_jwt_identity(),
        disease=disease,
        input_data=payload,
        prediction=label,
        probability=probability,
        model_used=model_used,
    )
    db.session.add(result)
    db.session.commit()

    body = result.to_dict()
    body["explanation"] = _explain_safely(disease, payload)
    return body


@predict_bp.get("/diseases")
def list_diseases():
    """Feature schema for every supported disease — drives the frontend forms."""
    out = {}
    for d in predictor.DISEASES:
        out[d] = {
            "features": predictor.get_feature_names(d),
            "metrics": {
                k: v for k, v in predictor.get_metrics(d).items()
                if k in ("accuracy", "f1", "roc_auc", "precision", "recall", "chosen_model")
            },
        }
    return jsonify(out), 200


@predict_bp.get("/diseases/info")
def diseases_info():
    """Educational content (overview/symptoms/risk factors/prevention) per disease —
    general public-health information, not the ML feature schema."""
    return jsonify(DISEASE_INFO), 200


@predict_bp.post("/predict/<disease>")
@jwt_required()
def predict_disease(disease):
    if disease not in predictor.DISEASES:
        return jsonify({"error": f"unknown disease '{disease}'"}), 404

    payload = request.get_json(silent=True) or {}
    invalid = _validate_or_none(disease, payload)
    if invalid:
        return invalid

    try:
        label, probability, model_used = predictor.predict(disease, payload)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"prediction failed: {e}"}), 400

    return jsonify(_save_result(disease, payload, label, probability, model_used)), 201


@predict_bp.post("/predict/<disease>/upload")
@jwt_required()
def predict_from_csv(disease):
    """Accepts a single-row CSV whose header matches (a subset of) the model's
    feature names and returns a prediction, same as the JSON endpoint."""
    if disease not in predictor.DISEASES:
        return jsonify({"error": f"unknown disease '{disease}'"}), 404

    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "upload a .csv file under the 'file' field"}), 400

    raw = file.read()
    # A real CSV is text — a null byte is a strong signal this is actually a
    # renamed binary file (exe, image, etc.) wearing a .csv extension.
    if b"\x00" in raw:
        return jsonify({"error": "file does not look like a valid CSV"}), 400

    try:
        text = raw.decode("utf-8-sig", errors="strict")
        reader = csv.DictReader(io.StringIO(text))
        row = next(reader, None)
    except UnicodeDecodeError:
        return jsonify({"error": "file is not valid UTF-8 text"}), 400
    except csv.Error:
        return jsonify({"error": "could not parse CSV file"}), 400

    if not row:
        return jsonify({"error": "CSV file has no data rows"}), 400

    payload = {k.strip(): v for k, v in row.items() if k}

    known_features = set(predictor.get_feature_names(disease))
    if not known_features.intersection(payload.keys()):
        return jsonify({
            "error": f"none of the CSV columns match known fields for '{disease}' "
                     f"— check the column headers against GET /api/diseases",
        }), 400

    invalid = _validate_or_none(disease, payload)
    if invalid:
        return invalid

    try:
        label, probability, model_used = predictor.predict(disease, payload)
    except Exception as e:
        return jsonify({"error": f"prediction failed: {e}"}), 400

    return jsonify(_save_result(disease, payload, label, probability, model_used)), 201


@predict_bp.post("/predict/<disease>/upload-pdf")
@jwt_required()
def predict_from_pdf(disease):
    """Best-effort: extracts known field values from a lab-report PDF's text
    and runs the same prediction pipeline as the CSV/JSON endpoints. PDF
    layouts vary widely, so not every field is guaranteed to be found —
    unmatched fields fall back to median imputation like everywhere else.
    Scanned/image-only PDFs (no text layer) aren't supported — there's no OCR
    step here."""
    if disease not in predictor.DISEASES:
        return jsonify({"error": f"unknown disease '{disease}'"}), 404

    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "upload a .pdf file under the 'file' field"}), 400

    raw = file.read()
    if not raw.startswith(b"%PDF-"):
        return jsonify({"error": "file does not look like a valid PDF"}), 400

    try:
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception:
        return jsonify({"error": "could not read PDF file"}), 400

    if not text.strip():
        return jsonify({"error": "no extractable text found in PDF (is it a scanned image?)"}), 400

    payload = extract_fields_from_pdf_text(text, disease)
    known_features = set(predictor.get_feature_names(disease))
    if not known_features.intersection(payload.keys()):
        return jsonify({
            "error": f"could not find any recognizable fields for '{disease}' in this PDF — "
                     f"try the manual form or a CSV export instead",
        }), 400

    invalid = _validate_or_none(disease, payload)
    if invalid:
        return invalid

    try:
        label, probability, model_used = predictor.predict(disease, payload)
    except Exception as e:
        return jsonify({"error": f"prediction failed: {e}"}), 400

    body = _save_result(disease, payload, label, probability, model_used)
    body["fields_found"] = sorted(payload.keys())
    body["fields_missing"] = sorted(known_features - payload.keys())
    return jsonify(body), 201


@predict_bp.get("/history")
@jwt_required()
def history():
    disease = request.args.get("disease")
    q = TestResult.query.filter_by(user_id=get_jwt_identity())
    if disease:
        q = q.filter_by(disease=disease)
    results = q.order_by(TestResult.created_at.desc()).all()
    return jsonify([r.to_dict() for r in results]), 200


@predict_bp.delete("/history/<result_id>")
@jwt_required()
def delete_history(result_id):
    result = TestResult.query.filter_by(id=result_id, user_id=get_jwt_identity()).first()
    if not result:
        return jsonify({"error": "not found"}), 404
    db.session.delete(result)
    db.session.commit()
    return jsonify({"deleted": True}), 200


@predict_bp.get("/history/<result_id>/export")
@jwt_required()
def export_result_pdf(result_id):
    result = TestResult.query.filter_by(id=result_id, user_id=get_jwt_identity()).first()
    if not result:
        return jsonify({"error": "not found"}), 404

    pdf_bytes = build_result_pdf(result, result.disease)
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"mdds-{result.disease}-{result.id[:8]}.pdf",
    )
