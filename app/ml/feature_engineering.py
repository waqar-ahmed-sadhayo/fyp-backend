"""Domain-informed derived features, added on top of the raw dataset columns.

These are real clinical ratios/indices, not arbitrary polynomial features —
picked because a clinician would actually compute them from the same raw
panel values. Shared between training (`train_models.py`, applied to the
full dataset) and inference (`predictor.py`, applied to one row at a time)
so the two can never drift apart.

Users only ever provide the *raw* fields (see `feature_names` in the saved
bundle) — these derived columns are computed automatically, never asked for
directly.
"""
import numpy as np
import pandas as pd

_DERIVED = {
    "heart": ["thalach_age_dev"],
    "diabetes": ["glucose_bmi", "homa_ir_proxy"],
    "liver": ["bilirubin_ratio", "ast_alt_ratio"],
}


def derived_feature_names(disease: str) -> list:
    return list(_DERIVED.get(disease, []))


def add_derived_features(df: pd.DataFrame, disease: str) -> pd.DataFrame:
    df = df.copy()

    if disease == "heart":
        # Deviation from the classic age-predicted max heart rate (220 - age).
        # A large negative value (achieved HR well below predicted max) is a
        # recognized marker of reduced cardiac functional capacity.
        df["thalach_age_dev"] = df["thalach"] - (220 - df["age"])

    elif disease == "diabetes":
        # Glucose x BMI interaction — metabolic risk compounds across both.
        df["glucose_bmi"] = df["Glucose"] * df["BMI"] / 1000
        # HOMA-IR: a real insulin-resistance index used in diabetes research,
        # (fasting glucose mg/dL x fasting insulin uU/mL) / 405.
        df["homa_ir_proxy"] = (df["Glucose"] * df["Insulin"]) / 405

    elif disease == "liver":
        # Direct/Total bilirubin ratio distinguishes hepatocellular from
        # obstructive liver injury patterns.
        df["bilirubin_ratio"] = df["Direct_Bilirubin"] / df["Total_Bilirubin"]
        # AST/ALT ("De Ritis") ratio — a standard clinical marker of liver
        # disease type/severity.
        df["ast_alt_ratio"] = df["Aspartate_Aminotransferase"] / df["Alamine_Aminotransferase"]
        derived_cols = ["bilirubin_ratio", "ast_alt_ratio"]
        df[derived_cols] = df[derived_cols].replace([np.inf, -np.inf], np.nan)

    return df
