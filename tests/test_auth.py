from conftest import register


def test_register_success(client):
    data = register(client, email="alice@example.com")
    assert data["token"]
    assert data["refresh_token"]
    assert data["user"]["email"] == "alice@example.com"
    assert data["user"]["email_verified"] is False
    assert data["verification_token"]


def test_register_missing_fields(client):
    res = client.post("/api/auth/register", json={"email": "a@b.com"})
    assert res.status_code == 400


def test_register_invalid_email(client):
    res = client.post("/api/auth/register", json={
        "full_name": "A", "email": "not-an-email", "password": "password123",
    })
    assert res.status_code == 400


def test_register_short_password(client):
    res = client.post("/api/auth/register", json={
        "full_name": "A", "email": "a@b.com", "password": "short",
    })
    assert res.status_code == 400


def test_register_duplicate_email(client):
    register(client, email="dupe@example.com")
    res = client.post("/api/auth/register", json={
        "full_name": "Dupe Two", "email": "dupe@example.com", "password": "password123",
    })
    assert res.status_code == 409


def test_login_success(client):
    register(client, email="login@example.com", password="password123")
    res = client.post("/api/auth/login", json={"email": "login@example.com", "password": "password123"})
    assert res.status_code == 200
    assert res.get_json()["token"]


def test_login_wrong_password(client):
    register(client, email="login2@example.com", password="password123")
    res = client.post("/api/auth/login", json={"email": "login2@example.com", "password": "wrongpass"})
    assert res.status_code == 401


def test_login_unknown_email(client):
    res = client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "password123"})
    assert res.status_code == 401


def test_me_requires_auth(client):
    res = client.get("/api/auth/me")
    assert res.status_code == 401


def test_me_returns_user(client, auth_headers):
    res = client.get("/api/auth/me", headers=auth_headers)
    assert res.status_code == 200
    assert res.get_json()["user"]["email"] == "test@example.com"


def test_verify_email_flow(client):
    data = register(client, email="verify@example.com")
    headers = {"Authorization": f"Bearer {data['token']}"}

    bad = client.post("/api/auth/verify-email", json={"token": "wrong-token"}, headers=headers)
    assert bad.status_code == 400

    good = client.post("/api/auth/verify-email", json={"token": data["verification_token"]}, headers=headers)
    assert good.status_code == 200
    assert good.get_json()["user"]["email_verified"] is True

    # verifying again is a no-op, not an error
    again = client.post("/api/auth/verify-email", json={"token": data["verification_token"]}, headers=headers)
    assert again.status_code == 200


def test_forgot_password_unknown_email_is_generic(client):
    res = client.post("/api/auth/forgot-password", json={"email": "nobody@example.com"})
    assert res.status_code == 200
    body = res.get_json()
    assert "reset_token" not in body


def test_forgot_and_reset_password_flow(client):
    register(client, email="reset@example.com", password="password123")

    forgot = client.post("/api/auth/forgot-password", json={"email": "reset@example.com"})
    assert forgot.status_code == 200
    token = forgot.get_json()["reset_token"]
    assert token

    reset = client.post("/api/auth/reset-password", json={"token": token, "password": "newpassword456"})
    assert reset.status_code == 200

    old_login = client.post("/api/auth/login", json={"email": "reset@example.com", "password": "password123"})
    assert old_login.status_code == 401

    new_login = client.post("/api/auth/login", json={"email": "reset@example.com", "password": "newpassword456"})
    assert new_login.status_code == 200


def test_reset_password_bad_token(client):
    res = client.post("/api/auth/reset-password", json={"token": "bogus", "password": "newpassword456"})
    assert res.status_code == 400


def test_reset_password_too_short(client):
    register(client, email="short@example.com")
    forgot = client.post("/api/auth/forgot-password", json={"email": "short@example.com"})
    token = forgot.get_json()["reset_token"]
    res = client.post("/api/auth/reset-password", json={"token": token, "password": "short"})
    assert res.status_code == 400


def test_refresh_issues_new_access_token(client):
    data = register(client, email="refresh@example.com")
    refresh_headers = {"Authorization": f"Bearer {data['refresh_token']}"}

    res = client.post("/api/auth/refresh", headers=refresh_headers)
    assert res.status_code == 200
    new_access_token = res.get_json()["token"]
    assert new_access_token

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {new_access_token}"})
    assert me.status_code == 200


def test_access_token_cannot_be_used_to_refresh(client):
    data = register(client, email="wrongtoken@example.com")
    res = client.post("/api/auth/refresh", headers={"Authorization": f"Bearer {data['token']}"})
    assert res.status_code == 422  # flask-jwt-extended rejects an access token where a refresh token is required


def test_logout_revokes_refresh_token(client):
    data = register(client, email="logout@example.com")
    refresh_headers = {"Authorization": f"Bearer {data['refresh_token']}"}

    out = client.post("/api/auth/logout", headers=refresh_headers)
    assert out.status_code == 200

    reused = client.post("/api/auth/refresh", headers=refresh_headers)
    assert reused.status_code == 401


def test_logout_requires_refresh_token(client):
    data = register(client, email="logout2@example.com")
    res = client.post("/api/auth/logout", headers={"Authorization": f"Bearer {data['token']}"})
    assert res.status_code == 422
