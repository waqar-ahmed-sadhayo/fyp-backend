import os
from datetime import timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _normalize_db_url(url):
    # Render (and some other providers) still hand out the legacy
    # "postgres://" scheme in DATABASE_URL; SQLAlchemy 1.4+ requires
    # "postgresql://" and raises on the old one. Rewrite it rather than
    # documenting a manual fix users will forget after every DB rotation.
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    return url


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-jwt-secret-change-me")
    # Access tokens are short-lived since they can't be individually revoked;
    # refresh tokens are long-lived but can be revoked on logout (see models.RevokedToken).
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)

    SQLALCHEMY_DATABASE_URI = _normalize_db_url(os.environ.get(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'mdds.db')}"
    ))
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")

    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB upload cap
    UPLOAD_EXTENSIONS = {".csv", ".pdf"}

    # Comma-separated emails that get is_admin=True on registration (or on
    # first login, if the account already existed). No admin UI/CLI yet —
    # this env var is the only way to grant the role. See phase 5 docs.
    ADMIN_EMAILS = {
        e.strip().lower()
        for e in os.environ.get("ADMIN_EMAILS", "").split(",")
        if e.strip()
    }

    # Comma-separated list of allowed frontend origins (e.g. your Vercel
    # deployment URL) for CORS on /api/*. Defaults to "*" so local dev keeps
    # working with no setup — set this explicitly in production instead of
    # relying on the wildcard.
    CORS_ORIGINS = [
        o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()
    ] or ["*"]
