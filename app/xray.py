import io

import pdfplumber
from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required
from PIL import Image, UnidentifiedImageError

from .ml import stone_predictor
from .models import TestResult, db

xray_bp = Blueprint("xray", __name__, url_prefix="/api")

ALLOWED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _save_result(source, filename, label, confidence, model_name):
    result = TestResult(
        user_id=get_jwt_identity(),
        disease="kidney_stone",
        # No raw image bytes stored — same "derived data only" pattern the
        # CSV/PDF lab-panel uploads already use (see predict.py).
        input_data={"source": source, "filename": filename[:200]},
        prediction=label,
        probability=confidence,
        model_used=model_name,
    )
    db.session.add(result)
    db.session.commit()
    return result.to_dict()


def _run_prediction(pil_image, source, filename):
    try:
        label, confidence, model_name = stone_predictor.predict_stone(pil_image)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"prediction failed: {e}"}), 400
    return jsonify(_save_result(source, filename, label, confidence, model_name)), 201


@xray_bp.get("/xray/kidney-stone/info")
def kidney_stone_info():
    """Metrics for the Dashboard/History card — same shape as GET /api/diseases's
    per-disease metrics block, so ScreeningCard can render it unchanged."""
    return jsonify({"metrics": stone_predictor.get_metrics()}), 200


@xray_bp.post("/xray/kidney-stone")
@jwt_required()
def predict_kidney_stone():
    """Accepts either a CT scan image (jpg/png) or a PDF containing one,
    under the 'file' field. PDFs are rasterized (page 1) rather than
    having an embedded image extracted — extraction is fragile across PDF
    producers, rendering the page always works."""
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "upload a file under the 'file' field"}), 400

    filename = file.filename
    lower = filename.lower()
    raw = file.read()

    if lower.endswith(".pdf"):
        if not raw.startswith(b"%PDF-"):
            return jsonify({"error": "file does not look like a valid PDF"}), 400
        try:
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                if not pdf.pages:
                    return jsonify({"error": "PDF has no pages"}), 400
                pil_image = pdf.pages[0].to_image(resolution=200).original
        except Exception:
            return jsonify({"error": "could not read PDF file"}), 400
        return _run_prediction(pil_image, "pdf", filename)

    if lower.endswith(ALLOWED_IMAGE_EXTENSIONS):
        try:
            pil_image = Image.open(io.BytesIO(raw))
            pil_image.load()  # forces full decode now, not lazily later
        except UnidentifiedImageError:
            return jsonify({"error": "file does not look like a valid image"}), 400
        except Exception:
            return jsonify({"error": "could not read image file"}), 400
        return _run_prediction(pil_image, "image", filename)

    return jsonify({
        "error": "upload a .jpg/.jpeg/.png image or a .pdf file",
    }), 400
