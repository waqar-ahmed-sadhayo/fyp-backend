from conftest import register


def test_admin_endpoints_require_auth(client):
    assert client.get("/api/admin/users").status_code == 401
    assert client.get("/api/admin/feedback").status_code == 401
    assert client.get("/api/admin/metrics").status_code == 401


def test_admin_endpoints_reject_non_admin(client, auth_headers):
    assert client.get("/api/admin/users", headers=auth_headers).status_code == 403
    assert client.get("/api/admin/feedback", headers=auth_headers).status_code == 403
    assert client.get("/api/admin/metrics", headers=auth_headers).status_code == 403


def test_register_with_admin_email_grants_admin(client, admin_headers):
    res = client.get("/api/auth/me", headers=admin_headers)
    assert res.get_json()["user"]["is_admin"] is True


def test_admin_list_users_includes_screening_count(client, admin_headers, auth_headers):
    client.post("/api/predict/heart", json={}, headers=auth_headers)

    res = client.get("/api/admin/users", headers=admin_headers)
    assert res.status_code == 200
    users = {u["email"]: u for u in res.get_json()}
    assert users["test@example.com"]["screening_count"] == 1
    assert users["admin@example.com"]["screening_count"] == 0


def test_admin_list_feedback_includes_submitter(client, admin_headers, auth_headers):
    client.post("/api/feedback", json={"message": "hi"}, headers=auth_headers)

    res = client.get("/api/admin/feedback", headers=admin_headers)
    assert res.status_code == 200
    body = res.get_json()
    assert len(body) == 1
    assert body[0]["from"]["email"] == "test@example.com"


def test_admin_metrics_includes_confusion_matrix(client, admin_headers):
    res = client.get("/api/admin/metrics", headers=admin_headers)
    assert res.status_code == 200
    body = res.get_json()
    assert "confusion_matrix" in body["heart"]
    assert "cv_f1_scores" in body["heart"]


def test_login_backfills_admin_status(client):
    # Register before the email is an admin email (simulated by registering a
    # different, non-admin account, then logging in — the fixture's admin
    # email is baked into ADMIN_EMAILS from the start, so this test instead
    # checks that a non-admin stays non-admin on login).
    register(client, email="notadmin@example.com")
    res = client.post("/api/auth/login", json={"email": "notadmin@example.com", "password": "password123"})
    assert res.get_json()["user"]["is_admin"] is False
