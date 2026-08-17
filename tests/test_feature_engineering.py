import numpy as np
import pandas as pd

from app.ml.feature_engineering import add_derived_features, derived_feature_names


def test_derived_feature_names_known_diseases():
    assert derived_feature_names("heart") == ["thalach_age_dev"]
    assert derived_feature_names("diabetes") == [
        "glucose_bmi", "homa_ir_proxy", "insulin_missing", "skin_thickness_missing",
    ]
    assert derived_feature_names("liver") == ["bilirubin_ratio", "ast_alt_ratio"]


def test_derived_feature_names_unknown_disease_is_empty():
    assert derived_feature_names("breast_cancer") == []
    assert derived_feature_names("kidney") == []
    assert derived_feature_names("not-a-real-disease") == []


def test_heart_thalach_age_dev():
    df = pd.DataFrame([{"age": 50, "thalach": 150}])
    out = add_derived_features(df, "heart")
    # 220 - 50 = 170 predicted max; 150 - 170 = -20
    assert out["thalach_age_dev"].iloc[0] == -20


def test_diabetes_derived_features():
    df = pd.DataFrame([{"Glucose": 120, "BMI": 30, "Insulin": 80, "SkinThickness": 25}])
    out = add_derived_features(df, "diabetes")
    assert out["glucose_bmi"].iloc[0] == 120 * 30 / 1000
    assert out["homa_ir_proxy"].iloc[0] == (120 * 80) / 405
    assert out["insulin_missing"].iloc[0] == 0.0
    assert out["skin_thickness_missing"].iloc[0] == 0.0


def test_diabetes_missing_indicator_flags():
    df = pd.DataFrame([{"Glucose": 120, "BMI": 30, "Insulin": np.nan, "SkinThickness": np.nan}])
    out = add_derived_features(df, "diabetes")
    assert out["insulin_missing"].iloc[0] == 1.0
    assert out["skin_thickness_missing"].iloc[0] == 1.0
    # homa_ir_proxy needs Insulin — stays NaN (median-imputed downstream),
    # not silently coerced to some fabricated number.
    assert np.isnan(out["homa_ir_proxy"].iloc[0])


def test_liver_derived_features():
    df = pd.DataFrame([{
        "Direct_Bilirubin": 0.5, "Total_Bilirubin": 1.0,
        "Aspartate_Aminotransferase": 40, "Alamine_Aminotransferase": 20,
    }])
    out = add_derived_features(df, "liver")
    assert out["bilirubin_ratio"].iloc[0] == 0.5
    assert out["ast_alt_ratio"].iloc[0] == 2.0


def test_liver_derived_features_handles_division_by_zero():
    df = pd.DataFrame([{
        "Direct_Bilirubin": 0.5, "Total_Bilirubin": 0.0,
        "Aspartate_Aminotransferase": 40, "Alamine_Aminotransferase": 0.0,
    }])
    out = add_derived_features(df, "liver")
    assert np.isnan(out["bilirubin_ratio"].iloc[0])
    assert np.isnan(out["ast_alt_ratio"].iloc[0])


def test_no_derived_features_for_breast_cancer_or_kidney():
    df = pd.DataFrame([{"a": 1, "b": 2}])
    out_bc = add_derived_features(df, "breast_cancer")
    out_kd = add_derived_features(df, "kidney")
    assert list(out_bc.columns) == ["a", "b"]
    assert list(out_kd.columns) == ["a", "b"]
