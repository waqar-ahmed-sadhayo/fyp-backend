def test_get_profile_requires_auth(client):
    res = client.get("/api/user/profile")
    assert res.status_code == 401


def test_get_profile(client, auth_headers):
    res = client.get("/api/user/profile", headers=auth_headers)
    assert res.status_code == 200
    assert res.get_json()["user"]["email"] == "test@example.com"


def test_update_profile(client, auth_headers):
    res = client.put("/api/user/profile", json={
        "full_name": "New Name", "age": 30, "gender": "female",
    }, headers=auth_headers)
    assert res.status_code == 200
    user = res.get_json()["user"]
    assert user["full_name"] == "New Name"
    assert user["age"] == 30
    assert user["gender"] == "female"


def test_update_profile_empty_name_rejected(client, auth_headers):
    res = client.put("/api/user/profile", json={"full_name": "  "}, headers=auth_headers)
    assert res.status_code == 400


def test_update_profile_invalid_age_rejected(client, auth_headers):
    res = client.put("/api/user/profile", json={"age": 200}, headers=auth_headers)
    assert res.status_code == 400

    res = client.put("/api/user/profile", json={"age": "not-a-number"}, headers=auth_headers)
    assert res.status_code == 400


def test_update_profile_invalid_gender_rejected(client, auth_headers):
    res = client.put("/api/user/profile", json={"gender": "robot"}, headers=auth_headers)
    assert res.status_code == 400


def test_update_profile_clears_age(client, auth_headers):
    client.put("/api/user/profile", json={"age": 40}, headers=auth_headers)
    res = client.put("/api/user/profile", json={"age": None}, headers=auth_headers)
    assert res.status_code == 200
    assert res.get_json()["user"]["age"] is None
