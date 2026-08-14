"""Best-effort extraction of known model features from a lab-report PDF's
text. Report layouts vary a lot, so this is intentionally simple — a regex
search per feature name, not a general-purpose document-understanding model.
Anything not found falls back to the predictor's existing median imputation,
same as a CSV upload with missing columns."""
import re

from .ml import predictor


def extract_fields_from_pdf_text(text: str, disease: str) -> dict:
    features = predictor.get_feature_names(disease)
    payload = {}

    for feature in features:
        variants = {feature, feature.replace("_", " "), feature.replace("_", " ").title()}
        for variant in variants:
            match = re.search(
                rf"{re.escape(variant)}\s*[:\-]?\s*([-+]?\d+(?:\.\d+)?)",
                text,
                re.IGNORECASE,
            )
            if match:
                payload[feature] = match.group(1)
                break

    return payload
