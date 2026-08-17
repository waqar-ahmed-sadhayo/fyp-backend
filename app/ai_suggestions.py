from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required

from .ai_assistant import AIAssistantError, contains_emergency_keywords, generate_response
from .ai_prompts import HEALTH_SUGGESTION_SYSTEM_PROMPT
from .disease_content import DISEASE_INFO
from .extensions import limiter
from .models import TestResult

ai_suggestions_bp = Blueprint("ai_suggestions", __name__, url_prefix="/api")


def _build_user_message(result: TestResult) -> str:
    disease_label = DISEASE_INFO.get(result.disease, {}).get("label", result.disease)
    lines = [
        f"Disease screened: {disease_label}",
        f"Model prediction: {result.prediction}",
        f"Confidence/probability: {result.probability * 100:.1f}%",
        "",
        "Input values used for this screening:",
    ]
    for key, value in (result.input_data or {}).items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


@ai_suggestions_bp.post("/health-suggestions")
@jwt_required()
@limiter.limit("10 per minute")
def health_suggestions():
    data = request.get_json(silent=True) or {}
    result_id = data.get("result_id")
    if not result_id:
        return jsonify({"error": "result_id is required"}), 400

    result = TestResult.query.filter_by(id=result_id, user_id=get_jwt_identity()).first()
    if not result:
        return jsonify({"error": "test result not found"}), 404

    user_message = _build_user_message(result)
    try:
        # The visible answer here is short (3-4 sentences) but gemini's
        # "thinking" tokens count against this same budget — see the note
        # in ai_assistant.generate_chat_response. 900 leaves comfortable
        # room for thinking + the actual answer without ballooning cost.
        suggestion = generate_response(HEALTH_SUGGESTION_SYSTEM_PROMPT, user_message, max_tokens=900)
    except AIAssistantError as e:
        return jsonify({"error": e.message}), e.status_code

    return jsonify({
        "result_id": result.id,
        "disease": result.disease,
        "suggestion": suggestion,
        "is_emergency": contains_emergency_keywords(suggestion),
    }), 200
