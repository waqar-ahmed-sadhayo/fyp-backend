import io

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def make_pdf_bytes(lines):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    for line in lines:
        c.drawString(50, y, line)
        y -= 20
    c.save()
    return buf.getvalue()


# ---- export ----

def test_export_requires_auth(client):
    res = client.get("/api/history/some-id/export")
    assert res.status_code == 401


def test_export_not_found(client, auth_headers):
    res = client.get("/api/history/does-not-exist/export", headers=auth_headers)
    assert res.status_code == 404


def test_export_pdf(client, auth_headers):
    created = client.post("/api/predict/heart", json={}, headers=auth_headers)
    result_id = created.get_json()["id"]

    res = client.get(f"/api/history/{result_id}/export", headers=auth_headers)
    assert res.status_code == 200
    assert res.content_type == "application/pdf"
    assert res.data.startswith(b"%PDF")


def test_export_not_owned_by_user(client, auth_headers):
    from conftest import register

    created = client.post("/api/predict/heart", json={}, headers=auth_headers)
    result_id = created.get_json()["id"]

    other = register(client, email="otherexport@example.com")
    other_headers = {"Authorization": f"Bearer {other['token']}"}

    res = client.get(f"/api/history/{result_id}/export", headers=other_headers)
    assert res.status_code == 404


# ---- upload-pdf ----

def test_upload_pdf_requires_auth(client):
    res = client.post("/api/predict/heart/upload-pdf", data={})
    assert res.status_code == 401


def test_upload_pdf_rejects_non_pdf_extension(client, auth_headers):
    fake = io.BytesIO(b"hello")
    res = client.post(
        "/api/predict/heart/upload-pdf",
        data={"file": (fake, "report.txt")},
        headers=auth_headers,
        content_type="multipart/form-data",
    )
    assert res.status_code == 400


def test_upload_pdf_rejects_invalid_pdf_content(client, auth_headers):
    fake = io.BytesIO(b"not actually a pdf")
    res = client.post(
        "/api/predict/heart/upload-pdf",
        data={"file": (fake, "report.pdf")},
        headers=auth_headers,
        content_type="multipart/form-data",
    )
    assert res.status_code == 400


def test_upload_pdf_extracts_recognized_fields_and_predicts(client, auth_headers):
    pdf_bytes = make_pdf_bytes(["Lab Report", "age: 52", "trestbps: 130", "chol: 210"])
    res = client.post(
        "/api/predict/heart/upload-pdf",
        data={"file": (io.BytesIO(pdf_bytes), "report.pdf")},
        headers=auth_headers,
        content_type="multipart/form-data",
    )
    assert res.status_code == 201
    body = res.get_json()
    assert body["disease"] == "heart"
    assert "age" in body["fields_found"]
    assert "trestbps" in body["fields_found"]
    assert body["input_data"]["age"] == "52"


def test_upload_pdf_no_recognizable_fields(client, auth_headers):
    pdf_bytes = make_pdf_bytes(["Just some unrelated text", "nothing useful here"])
    res = client.post(
        "/api/predict/heart/upload-pdf",
        data={"file": (io.BytesIO(pdf_bytes), "report.pdf")},
        headers=auth_headers,
        content_type="multipart/form-data",
    )
    assert res.status_code == 400
