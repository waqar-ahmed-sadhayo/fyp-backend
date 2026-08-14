import pytest

from app import create_app
from app.models import db
from conftest import TestConfig, register


@pytest.fixture()
def limited_client(tmp_path):
    config = TestConfig(f"sqlite:///{tmp_path / 'ratelimit.db'}")
    config.RATELIMIT_ENABLED = True
    flask_app = create_app(config)
    with flask_app.test_client() as client:
        yield client
    with flask_app.app_context():
        db.session.remove()
        db.drop_all()


def test_login_is_rate_limited(limited_client):
    register(limited_client, email="ratelimit@example.com", password="password123")

    # limit is 10/minute; the 11th attempt in the same window should be throttled
    statuses = []
    for _ in range(11):
        res = limited_client.post("/api/auth/login", json={
            "email": "ratelimit@example.com", "password": "wrong-password",
        })
        statuses.append(res.status_code)

    assert statuses[:10] == [401] * 10
    assert statuses[10] == 429
