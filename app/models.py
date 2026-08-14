import uuid
from datetime import datetime, timezone

import bcrypt
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def _uuid():
    return str(uuid.uuid4())


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.String(36), primary_key=True, default=_uuid)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.LargeBinary, nullable=False)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Email verification and password reset use tokens issued directly to the
    # caller (see auth.py) rather than a real mail transport — there's no SMTP
    # service configured in this environment. Swap that piece out before any
    # public deployment; the token/expiry plumbing here doesn't need to change.
    email_verified = db.Column(db.Boolean, default=False, nullable=False)
    verification_token = db.Column(db.String(64), nullable=True)
    reset_token = db.Column(db.String(64), nullable=True)
    reset_token_expiry = db.Column(db.DateTime, nullable=True)

    is_admin = db.Column(db.Boolean, default=False, nullable=False)

    results = db.relationship("TestResult", backref="user", lazy=True,
                               cascade="all, delete-orphan")

    def set_password(self, raw_password: str):
        self.password_hash = bcrypt.hashpw(raw_password.encode("utf-8"), bcrypt.gensalt())

    def check_password(self, raw_password: str) -> bool:
        return bcrypt.checkpw(raw_password.encode("utf-8"), self.password_hash)

    def to_dict(self):
        return {
            "id": self.id,
            "full_name": self.full_name,
            "email": self.email,
            "age": self.age,
            "gender": self.gender,
            "email_verified": self.email_verified,
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat(),
        }


class TestResult(db.Model):
    __tablename__ = "test_results"

    id = db.Column(db.String(36), primary_key=True, default=_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    disease = db.Column(db.String(50), nullable=False)
    input_data = db.Column(db.JSON, nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    model_used = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "disease": self.disease,
            "input_data": self.input_data,
            "prediction": self.prediction,
            "probability": self.probability,
            "model_used": self.model_used,
            "created_at": self.created_at.isoformat(),
        }


class Feedback(db.Model):
    __tablename__ = "feedback"

    id = db.Column(db.String(36), primary_key=True, default=_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    subject = db.Column(db.String(200), nullable=True)
    message = db.Column(db.String(4000), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "subject": self.subject,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
        }


class RevokedToken(db.Model):
    """JWT jti blocklist. Only refresh tokens are ever added here (on logout) —
    access tokens are short-lived instead of individually revocable; see
    phases/backend/phase-4-testing-security.md for that tradeoff."""
    __tablename__ = "revoked_tokens"

    jti = db.Column(db.String(36), primary_key=True)
    revoked_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
