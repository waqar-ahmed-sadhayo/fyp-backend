import pytest

from app import create_app
from app.config import Config
from app.models import db


ADMIN_EMAIL = "admin@example.com"


class TestConfig(Config):
    def __init__(self, db_uri):
        self.TESTING = True
        self.SECRET_KEY = "test-secret"
        self.JWT_SECRET_KEY = "test-jwt-secret"
        self.SQLALCHEMY_DATABASE_URI = db_uri
        self.RATELIMIT_ENABLED = False
        self.ADMIN_EMAILS = {ADMIN_EMAIL}


@pytest.fixture()
def app(tmp_path):
    flask_app = create_app(TestConfig(f"sqlite:///{tmp_path / 'test.db'}"))
    yield flask_app
    with flask_app.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def register(client, email="test@example.com", password="password123", full_name="Test User"):
    res = client.post("/api/auth/register", json={
        "full_name": full_name, "email": email, "password": password,
    })
    assert res.status_code == 201, res.get_json()
    return res.get_json()


@pytest.fixture()
def registered_user(client):
    return register(client)


@pytest.fixture()
def auth_headers(registered_user):
    return {"Authorization": f"Bearer {registered_user['token']}"}


@pytest.fixture()
def admin_headers(client):
    data = register(client, email=ADMIN_EMAIL, full_name="Admin User")
    assert data["user"]["is_admin"] is True
    return {"Authorization": f"Bearer {data['token']}"}
