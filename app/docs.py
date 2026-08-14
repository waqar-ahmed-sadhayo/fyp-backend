"""Serves the OpenAPI spec + a Swagger UI page at /api/docs."""
import os

from flask import Blueprint, send_from_directory
from flask_swagger_ui import get_swaggerui_blueprint

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_FILENAME = "openapi.yaml"

spec_bp = Blueprint("spec", __name__)


@spec_bp.get("/api/openapi.yaml")
def openapi_spec():
    return send_from_directory(REPO_ROOT, SPEC_FILENAME, mimetype="text/yaml")


swagger_ui_bp = get_swaggerui_blueprint(
    "/api/docs",
    "/api/openapi.yaml",
    config={"app_name": "Multi-Disease Detection System API"},
)
