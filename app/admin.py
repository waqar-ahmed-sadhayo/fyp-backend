from functools import wraps

from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, verify_jwt_in_request

from .ml import predictor
from .models import Feedback, TestResult, User, db

admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")


def admin_required(fn):
    """Auth + role check in one decorator, rather than stacking @jwt_required()
    under a separate role check — keeps the 401-vs-403 distinction simple:
    no/invalid token -> 401 (flask-jwt-extended's default), valid token but
    not an admin -> 403."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        verify_jwt_in_request()
        user = db.session.get(User, get_jwt_identity())
        if not user or not user.is_admin:
            return jsonify({"error": "admin access required"}), 403
        return fn(*args, **kwargs)
    return wrapper


@admin_bp.get("/users")
@admin_required
def list_users():
    users = User.query.order_by(User.created_at.desc()).all()
    out = []
    for u in users:
        d = u.to_dict()
        d["screening_count"] = TestResult.query.filter_by(user_id=u.id).count()
        out.append(d)
    return jsonify(out), 200


@admin_bp.patch("/users/<user_id>/admin")
@admin_required
def set_user_admin(user_id):
    """Grants or revokes admin access for a user — lets an admin promote
    someone through the UI instead of needing the ADMIN_EMAILS env var and a
    server restart every time. No self-demotion guard: if you lock yourself
    out, ADMIN_EMAILS + a login still recovers access (see auth.py's
    backfill), so an extra safety check here would just be redundant."""
    payload = request.get_json(silent=True) or {}
    is_admin = payload.get("is_admin")
    if not isinstance(is_admin, bool):
        return jsonify({"error": "'is_admin' (boolean) is required"}), 400

    target = db.session.get(User, user_id)
    if not target:
        return jsonify({"error": "user not found"}), 404

    target.is_admin = is_admin
    db.session.commit()
    return jsonify(target.to_dict()), 200


@admin_bp.delete("/users/<user_id>")
@admin_required
def delete_user(user_id):
    """Permanently deletes an account and everything tied to it (screening
    history, feedback — both cascade via the User model's relationships).
    Unlike set_user_admin, self-service has no recovery path here (no
    ADMIN_EMAILS backfill for a deleted row), so deleting your own account
    is blocked rather than just discouraged."""
    admin_id = get_jwt_identity()
    if user_id == admin_id:
        return jsonify({"error": "cannot delete your own account"}), 400

    target = db.session.get(User, user_id)
    if not target:
        return jsonify({"error": "user not found"}), 404

    db.session.delete(target)
    db.session.commit()
    return "", 204


@admin_bp.get("/feedback")
@admin_required
def list_feedback():
    items = Feedback.query.order_by(Feedback.created_at.desc()).all()
    out = []
    for f in items:
        submitter = db.session.get(User, f.user_id)
        d = f.to_dict()
        d["from"] = {"email": submitter.email, "full_name": submitter.full_name} if submitter else None
        out.append(d)
    return jsonify(out), 200


@admin_bp.get("/metrics")
@admin_required
def model_metrics():
    """Full per-disease metrics (confusion matrix, per-algorithm CV scores) for
    monitoring — the public GET /api/diseases only exposes a filtered subset."""
    return jsonify({d: predictor.get_metrics(d) for d in predictor.DISEASES}), 200
