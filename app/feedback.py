from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required

from .models import Feedback, db

feedback_bp = Blueprint("feedback", __name__, url_prefix="/api")


@feedback_bp.post("/feedback")
@jwt_required()
def submit_feedback():
    data = request.get_json(silent=True) or {}
    subject = (data.get("subject") or "").strip()
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "message is required"}), 400
    if len(message) > 4000:
        return jsonify({"error": "message too long (max 4000 characters)"}), 400

    fb = Feedback(user_id=get_jwt_identity(), subject=subject[:200] or None, message=message)
    db.session.add(fb)
    db.session.commit()
    return jsonify(fb.to_dict()), 201
