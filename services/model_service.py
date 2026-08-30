from pathlib import Path

import joblib
import numpy as np
import shap


# Project root: E:\Resume-Screener-validation
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Load each saved artifact once when this module is imported.
category_model = joblib.load(PROJECT_ROOT / "category_classifier.pkl")
suitability_model = joblib.load(PROJECT_ROOT / "suitability_model.pkl")
category_vectorizer = joblib.load(PROJECT_ROOT / "tfidf_vectorizer.pkl")
suitability_vectorizer = joblib.load(PROJECT_ROOT / "suitability_vectorizer.pkl")
skill_vectorizer = joblib.load(PROJECT_ROOT / "skill_vectorizer.pkl")


# The suitability model is a GradientBoostingRegressor.
explainer = shap.TreeExplainer(suitability_model)


def predict_category(resume_text):
    """Predict the resume category using the saved TF-IDF pipeline."""
    category_matrix = category_vectorizer.transform([resume_text])
    category = category_model.predict(category_matrix)[0]
    probabilities = category_model.predict_proba(category_matrix)[0]

    return str(category), probabilities


def predict_suitability(features_df):
    """Predict raw and display suitability scores from the eight features."""
    raw_score = float(suitability_model.predict(features_df)[0])
    display_score = float(np.clip(raw_score, 0, 100))

    return raw_score, display_score


def explain_suitability(features_df, feature_names):
    """Return SHAP impacts for one row of validated features."""
    shap_values = explainer.shap_values(features_df)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_row = np.asarray(shap_values)[0]
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    reasons = []
    for feature_name, feature_value, impact in zip(
        feature_names,
        features_df.iloc[0].values,
        shap_row,
    ):
        impact = float(impact)
        reasons.append(
            {
                "feature": feature_name,
                "value": float(feature_value),
                "impact": impact,
                "direction": "increases" if impact > 0 else "decreases",
            }
        )

    reconstructed_score = float(base_value + shap_row.sum())

    return {
        "base_value": base_value,
        "reconstructed_score": reconstructed_score,
        "reasons": reasons,
    }
