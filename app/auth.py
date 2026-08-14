import re
import secrets
from datetime import datetime, timedelta

from flask import Blueprint, current_app, jsonify, request
from flask_jwt_extended import (create_access_token, create_refresh_token,
                                 get_jwt, get_jwt_identity, jwt_required)

from .extensions import limiter
from .models import RevokedToken, User, db

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _tokens_for(user):
    return {"token": create_access_token(identity=user.id),
            "refresh_token": create_refresh_token(identity=user.id)}

# No SMTP/mail service is configured in this environment, so verification and
# reset tokens are handed back directly in the API response instead of being
# emailed. Anything under a "dev_note" key is a stand-in for real email
# delivery and should be removed once that's wired up (see deployment phase).
RESET_TOKEN_TTL = timedelta(hours=1)


@auth_bp.post("/register")
@limiter.limit("5 per minute")
def register():
    data = request.get_json(silent=True) or {}
    full_name = (data.get("full_name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not full_name or not email or not password:
        return jsonify({"error": "full_name, email and password are required"}), 400
    if not EMAIL_RE.match(email):
        return jsonify({"error": "invalid email address"}), 400
    if len(password) < 8:
        return jsonify({"error": "password must be at least 8 characters"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "an account with this email already exists"}), 409

    user = User(full_name=full_name, email=email)
    user.set_password(password)
    user.verification_token = secrets.token_urlsafe(24)
    user.is_admin = email in current_app.config["ADMIN_EMAILS"]
    db.session.add(user)
    db.session.commit()

    return jsonify({
        **_tokens_for(user),
        "user": user.to_dict(),
        "verification_token": user.verification_token,
        "dev_note": "No email service is configured — verify with this token via "
                    "POST /api/auth/verify-email instead of an emailed link.",
    }), 201


@auth_bp.post("/login")
@limiter.limit("10 per minute")
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({"error": "invalid email or password"}), 401

    # Backfill admin status if this email was added to ADMIN_EMAILS after the
    # account already existed — there's no admin UI/CLI to promote otherwise.
    should_be_admin = email in current_app.config["ADMIN_EMAILS"]
    if should_be_admin != user.is_admin:
        user.is_admin = should_be_admin
        db.session.commit()

    return jsonify({**_tokens_for(user), "user": user.to_dict()}), 200


@auth_bp.post("/refresh")
@jwt_required(refresh=True)
def refresh():
    return jsonify({"token": create_access_token(identity=get_jwt_identity())}), 200


@auth_bp.post("/logout")
@jwt_required(refresh=True)
def logout():
    """Revokes the refresh token used to call this endpoint. The short-lived
    access token issued alongside it stays valid until it naturally expires —
    see the RevokedToken model docstring for that tradeoff."""
    jti = get_jwt()["jti"]
    db.session.add(RevokedToken(jti=jti))
    db.session.commit()
    return jsonify({"message": "logged out"}), 200


@auth_bp.get("/me")
@jwt_required()
def me():
    user = User.query.get(get_jwt_identity())
    if not user:
        return jsonify({"error": "user not found"}), 404
    return jsonify({"user": user.to_dict()}), 200


@auth_bp.post("/verify-email")
@jwt_required()
def verify_email():
    user = User.query.get(get_jwt_identity())
    if not user:
        return jsonify({"error": "user not found"}), 404
    if user.email_verified:
        return jsonify({"message": "already verified", "user": user.to_dict()}), 200

    data = request.get_json(silent=True) or {}
    token = data.get("token") or ""
    if not token or token != user.verification_token:
        return jsonify({"error": "invalid verification token"}), 400

    user.email_verified = True
    user.verification_token = None
    db.session.commit()
    return jsonify({"message": "email verified", "user": user.to_dict()}), 200


@auth_bp.post("/forgot-password")
@limiter.limit("5 per minute")
def forgot_password():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()

    # Always return the same shape whether or not the account exists, so the
    # response alone can't be used to enumerate registered emails.
    generic = {"message": "if an account exists for this email, a reset token has been issued"}

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify(generic), 200

    user.reset_token = secrets.token_urlsafe(24)
    user.reset_token_expiry = datetime.utcnow() + RESET_TOKEN_TTL
    db.session.commit()

    return jsonify({
        **generic,
        "reset_token": user.reset_token,
        "dev_note": "No email service is configured — use this token with "
                    "POST /api/auth/reset-password instead of an emailed link. "
                    "Expires in 1 hour.",
    }), 200


@auth_bp.post("/reset-password")
@limiter.limit("10 per minute")
def reset_password():
    data = request.get_json(silent=True) or {}
    token = data.get("token") or ""
    new_password = data.get("password") or ""

    if len(new_password) < 8:
        return jsonify({"error": "password must be at least 8 characters"}), 400

    user = User.query.filter_by(reset_token=token).first() if token else None
    if not user or not user.reset_token_expiry or user.reset_token_expiry < datetime.utcnow():
        return jsonify({"error": "invalid or expired reset token"}), 400

    user.set_password(new_password)
    user.reset_token = None
    user.reset_token_expiry = None
    db.session.commit()
    return jsonify({"message": "password reset successful"}), 200
