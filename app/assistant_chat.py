from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required

from .ai_assistant import AIAssistantError, contains_emergency_keywords, generate_chat_response
from .ai_prompts import WEBSITE_ASSISTANT_SYSTEM_PROMPT
from .extensions import limiter

assistant_chat_bp = Blueprint("assistant_chat", __name__, url_prefix="/api")

# Bounds on what the client can send, so one chat session can't balloon the
# token bill — the widget only keeps a short rolling window anyway, but the
# backend enforces this independently rather than trusting the frontend.
MAX_MESSAGES = 20
MAX_MESSAGE_LENGTH = 4000


def _validate_messages(raw):
    if not isinstance(raw, list) or not raw:
        return None, "messages must be a non-empty list"
    if len(raw) > MAX_MESSAGES:
        raw = raw[-MAX_MESSAGES:]

    cleaned = []
    for item in raw:
        if not isinstance(item, dict):
            return None, "each message must be an object"
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant"):
            return None, "message role must be 'user' or 'assistant'"
        if not isinstance(content, str) or not content.strip():
            return None, "message content must be non-empty text"
        cleaned.append({"role": role, "content": content.strip()[:MAX_MESSAGE_LENGTH]})

    if cleaned[-1]["role"] != "user":
        return None, "the last message must be from the user"
    return cleaned, None


@assistant_chat_bp.post("/assistant/chat")
@jwt_required()
@limiter.limit("20 per minute")
def assistant_chat():
    data = request.get_json(silent=True) or {}
    messages, error = _validate_messages(data.get("messages"))
    if error:
        return jsonify({"error": error}), 400

    try:
        reply = generate_chat_response(WEBSITE_ASSISTANT_SYSTEM_PROMPT, messages)
    except AIAssistantError as e:
        return jsonify({"error": e.message}), e.status_code

    return jsonify({
        "reply": reply,
        "is_emergency": contains_emergency_keywords(reply, messages[-1]["content"]),
    }), 200
