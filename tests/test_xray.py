import io

from reportlab.pdfgen import canvas
from PIL import Image


def _png_bytes(size=(64, 64), color=(120, 120, 120)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _pdf_bytes():
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 700, "test scan report")
    c.save()
    buf.seek(0)
    return buf.read()


def test_kidney_stone_requires_auth(client):
    data = {"file": (io.BytesIO(_png_bytes()), "scan.png")}
    res = client.post("/api/xray/kidney-stone", data=data, content_type="multipart/form-data")
    assert res.status_code == 401


def test_kidney_stone_missing_file_rejected(client, auth_headers):
    res = client.post("/api/xray/kidney-stone", data={}, content_type="multipart/form-data", headers=auth_headers)
    assert res.status_code == 400


def test_kidney_stone_rejects_bad_extension(client, auth_headers):
    data = {"file": (io.BytesIO(b"not an image"), "scan.txt")}
    res = client.post("/api/xray/kidney-stone", data=data, content_type="multipart/form-data", headers=auth_headers)
    assert res.status_code == 400


def test_kidney_stone_rejects_corrupt_image(client, auth_headers):
    data = {"file": (io.BytesIO(b"not really a png"), "scan.png")}
    res = client.post("/api/xray/kidney-stone", data=data, content_type="multipart/form-data", headers=auth_headers)
    assert res.status_code == 400


def test_kidney_stone_image_upload_happy_path(client, auth_headers):
    data = {"file": (io.BytesIO(_png_bytes()), "scan.png")}
    res = client.post("/api/xray/kidney-stone", data=data, content_type="multipart/form-data", headers=auth_headers)
    assert res.status_code == 201, res.get_json()
    body = res.get_json()
    assert body["disease"] == "kidney_stone"
    assert body["prediction"] in ("stone", "no_stone")
    assert 0.0 <= body["probability"] <= 1.0
    assert body["input_data"]["source"] == "image"
    assert body["input_data"]["filename"] == "scan.png"

    hist = client.get("/api/history", headers=auth_headers)
    assert any(r["id"] == body["id"] for r in hist.get_json())


def test_kidney_stone_pdf_upload_happy_path(client, auth_headers):
    data = {"file": (io.BytesIO(_pdf_bytes()), "report.pdf")}
    res = client.post("/api/xray/kidney-stone", data=data, content_type="multipart/form-data", headers=auth_headers)
    assert res.status_code == 201, res.get_json()
    body = res.get_json()
    assert body["input_data"]["source"] == "pdf"


def test_kidney_stone_rejects_fake_pdf(client, auth_headers):
    data = {"file": (io.BytesIO(b"not a real pdf"), "report.pdf")}
    res = client.post("/api/xray/kidney-stone", data=data, content_type="multipart/form-data", headers=auth_headers)
    assert res.status_code == 400
