from flask import Flask, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager

from .config import Config
from .extensions import limiter
from .models import RevokedToken, db


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    jwt = JWTManager(app)
    CORS(app, resources={r"/api/*": {"origins": app.config["CORS_ORIGINS"]}})
    limiter.init_app(app)

    @jwt.token_in_blocklist_loader
    def is_token_revoked(_jwt_header, jwt_payload):
        return db.session.get(RevokedToken, jwt_payload["jti"]) is not None

    from .admin import admin_bp
    from .ai_suggestions import ai_suggestions_bp
    from .assistant_chat import assistant_chat_bp
    from .auth import auth_bp
    from .docs import spec_bp, swagger_ui_bp
    from .feedback import feedback_bp
    from .ml import predictor as ml_predictor
    from .predict import predict_bp
    from .user import user_bp

    try:
        ml_predictor.load_all()
    except Exception as e:
        raise RuntimeError(f"Failed to load ML models at startup: {e}") from e

    app.register_blueprint(auth_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(user_bp)
    app.register_blueprint(feedback_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(ai_suggestions_bp)
    app.register_blueprint(assistant_chat_bp)
    app.register_blueprint(spec_bp)
    app.register_blueprint(swagger_ui_bp)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"}), 200

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "file too large (max 5MB)"}), 413

    @app.errorhandler(429)
    def rate_limited(e):
        return jsonify({"error": "too many requests, please slow down"}), 429

    with app.app_context():
        db.create_all()

    return app
