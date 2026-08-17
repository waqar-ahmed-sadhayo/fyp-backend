"""Loads the ONNX kidney-stone CT model once and exposes predict_stone().
Mirrors predictor.py's lazy-load + load_all() shape, but this model is an
image classifier (onnxruntime), not a tabular scikit-learn bundle — kept
as a separate module rather than shoehorned into predictor.py's
tabular-specific pipeline (imputer/scaler/feature_engineering don't apply
to images)."""
import json
import logging
import os

import numpy as np
import onnxruntime as ort

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
ONNX_PATH = os.path.join(MODEL_DIR, "kidney_stone.onnx")
META_PATH = os.path.join(MODEL_DIR, "kidney_stone_meta.json")

logger = logging.getLogger("mdds.predict")

_session = None
_meta = None


def _load():
    global _session, _meta
    if _session is None:
        if not os.path.exists(ONNX_PATH) or not os.path.exists(META_PATH):
            raise FileNotFoundError(
                "No trained kidney-stone model found. Run "
                "`python -m app.ml.train_stone_model` first."
            )
        _session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        with open(META_PATH) as f:
            _meta = json.load(f)
    return _session, _meta


def load_all():
    """Eagerly loads the model bundle. Call this once at app startup (see
    app/__init__.py) — same "fail loudly at boot" reasoning as
    predictor.load_all()."""
    _load()


def get_metrics():
    return _load()[1]["metrics"]


def _softmax(logits):
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def preprocess(pil_image):
    """pil_image: a PIL.Image (already RGB). Returns a (1, 3, H, W) float32
    array normalized the same way training normalized it (see meta)."""
    _, meta = _load()
    size = meta["input_size"]
    mean = np.array(meta["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(meta["std"], dtype=np.float32).reshape(3, 1, 1)

    img = pil_image.convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # H, W, 3
    arr = arr.transpose(2, 0, 1)  # 3, H, W
    arr = (arr - mean) / std
    return arr[np.newaxis, ...].astype(np.float32)


def predict_stone(pil_image):
    """Returns (label:str, probability:float, model_name:str). label is
    "stone" or "no_stone" (never "borderline" — the tabular models' near-
    threshold ambiguity handling doesn't have an established equivalent
    for a single-image classifier yet; a wide margin, not exact 0.5, is
    what tips a stone call either way here)."""
    session, meta = _load()
    x = preprocess(pil_image)
    logits = session.run(["logits"], {"input": x})[0]
    probs = _softmax(logits)[0]
    stone_proba = float(probs[1])
    threshold = meta["threshold"]
    pred_class = int(stone_proba >= threshold)
    label = meta["label_map"][str(pred_class)]
    confidence = stone_proba if pred_class == 1 else float(probs[0])
    model_name = meta["metrics"]["chosen_model"]

    logger.info(
        "predict kidney_stone model=%s threshold=%.4f stone_proba=%.4f label=%s",
        model_name, threshold, stone_proba, label,
    )
    return label, round(confidence, 4), model_name
