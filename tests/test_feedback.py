def test_feedback_requires_auth(client):
    res = client.post("/api/feedback", json={"message": "hello"})
    assert res.status_code == 401


def test_submit_feedback(client, auth_headers):
    res = client.post("/api/feedback", json={
        "subject": "Bug report", "message": "The gauge flickers on load.",
    }, headers=auth_headers)
    assert res.status_code == 201
    body = res.get_json()
    assert body["subject"] == "Bug report"
    assert body["message"] == "The gauge flickers on load."


def test_feedback_requires_message(client, auth_headers):
    res = client.post("/api/feedback", json={"subject": "No message"}, headers=auth_headers)
    assert res.status_code == 400


def test_feedback_message_too_long_rejected(client, auth_headers):
    res = client.post("/api/feedback", json={"message": "x" * 4001}, headers=auth_headers)
    assert res.status_code == 400


def test_feedback_subject_is_optional(client, auth_headers):
    res = client.post("/api/feedback", json={"message": "No subject given"}, headers=auth_headers)
    assert res.status_code == 201
    assert res.get_json()["subject"] is None
