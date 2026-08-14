from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required

from .models import User, db

user_bp = Blueprint("user", __name__, url_prefix="/api/user")

VALID_GENDERS = {"male", "female", "other"}


@user_bp.get("/profile")
@jwt_required()
def get_profile():
    user = User.query.get(get_jwt_identity())
    if not user:
        return jsonify({"error": "user not found"}), 404
    return jsonify({"user": user.to_dict()}), 200


@user_bp.put("/profile")
@jwt_required()
def update_profile():
    user = User.query.get(get_jwt_identity())
    if not user:
        return jsonify({"error": "user not found"}), 404

    data = request.get_json(silent=True) or {}

    if "full_name" in data:
        full_name = (data.get("full_name") or "").strip()
        if not full_name:
            return jsonify({"error": "full_name cannot be empty"}), 400
        user.full_name = full_name

    if "age" in data:
        age = data.get("age")
        if age in (None, ""):
            user.age = None
        else:
            try:
                age = int(age)
            except (TypeError, ValueError):
                return jsonify({"error": "age must be a number"}), 400
            if age < 1 or age > 120:
                return jsonify({"error": "age must be between 1 and 120"}), 400
            user.age = age

    if "gender" in data:
        gender = (data.get("gender") or "").strip().lower() or None
        if gender and gender not in VALID_GENDERS:
            return jsonify({"error": "gender must be one of: male, female, other"}), 400
        user.gender = gender

    db.session.commit()
    return jsonify({"user": user.to_dict()}), 200
