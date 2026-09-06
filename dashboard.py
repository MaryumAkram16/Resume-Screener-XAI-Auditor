import numpy as np
import pandas as pd
import streamlit as st

from services.feature_service import FEATURE_COLS, build_features
from services.model_service import (
    explain_suitability,
    predict_category,
    predict_suitability,
)


st.set_page_config(
    page_title="Resume Screener XAI Auditor",
    page_icon="📄",
    layout="wide",
)


STRONG_RESUME = """Software engineer with experience in Python, SQL, machine learning,
and backend application development. Worked on data pipelines and APIs."""

STRONG_JOB = """We are seeking a backend engineer with Python, SQL, REST APIs,
and machine learning experience."""


st.title("Resume Screener XAI Auditor")
st.caption("Explore the saved Resume Screener prediction and the feature-level SHAP explanation.")

with st.sidebar:
    st.header("Input")
    load_sample = st.checkbox("Load strong-match sample", value=True)

    if load_sample:
        default_resume = STRONG_RESUME
        default_job = STRONG_JOB
    else:
        default_resume = ""
        default_job = ""

    st.info("Enter a resume and a job description, then click Analyze.")

resume_text = st.text_area(
    "Resume text",
    value=default_resume,
    height=240,
    placeholder="Paste the resume text here...",
)

job_text = st.text_area(
    "Job description",
    value=default_job,
    height=220,
    placeholder="Paste the job description here...",
)

analyze = st.button("Analyze resume", type="primary", use_container_width=True)


if analyze:
    if not resume_text.strip() or not job_text.strip():
        st.error("Both resume text and job description are required.")
        st.stop()

    with st.spinner("Calculating prediction and SHAP explanation..."):
        try:
            features_df, matched_skills = build_features(
                resume_text.strip(),
                job_text.strip(),
            )
            category, probabilities = predict_category(resume_text.strip())
            raw_score, display_score = predict_suitability(features_df)
            shap_result = explain_suitability(features_df, FEATURE_COLS)
        except Exception as error:
            st.error(f"The screening pipeline failed: {error}")
            st.stop()

    reasons = pd.DataFrame(shap_result["reasons"])
    reasons["absolute_impact"] = reasons["impact"].abs()
    reasons = reasons.sort_values("absolute_impact", ascending=False)

    st.success("Analysis completed successfully.")

    score_col, category_col, match_col = st.columns(3)
    with score_col:
        st.metric("Suitability score", f"{display_score:.2f} / 100")
    with category_col:
        st.metric("Predicted category", category)
        st.caption(f"Category probability: {max(probabilities):.2%}")
    with match_col:
        st.metric("Matched skills", len(matched_skills))
        st.caption(", ".join(matched_skills) if matched_skills else "No shared vocabulary skills")

    st.divider()

    tab_explanation, tab_features, tab_details = st.tabs(
        ["Why this score?", "Feature values", "Technical details"]
    )

    with tab_explanation:
        st.subheader("Feature impacts")
        st.write(
            "Positive values pushed the model score higher. Negative values pushed it lower."
        )

        chart_data = reasons.set_index("feature")["impact"]
        st.bar_chart(chart_data, horizontal=True)

        display_reasons = reasons[
            ["feature", "value", "impact", "direction"]
        ].copy()
        display_reasons["value"] = display_reasons["value"].map(lambda value: f"{value:.6f}")
        display_reasons["impact"] = display_reasons["impact"].map(lambda value: f"{value:+.6f}")
        st.dataframe(display_reasons, use_container_width=True, hide_index=True)

    with tab_features:
        st.subheader("Validated model features")
        st.dataframe(features_df, use_container_width=True, hide_index=True)
        st.caption("The feature order is preserved exactly as expected by the saved suitability model.")

    with tab_details:
        st.subheader("Prediction details")
        details = pd.DataFrame(
            {
                "item": [
                    "Raw suitability score",
                    "Display suitability score",
                    "SHAP base value",
                    "Reconstructed score",
                    "Reconstruction difference",
                ],
                "value": [
                    raw_score,
                    display_score,
                    shap_result["base_value"],
                    shap_result["reconstructed_score"],
                    abs(raw_score - shap_result["reconstructed_score"]),
                ],
            }
        )
        st.dataframe(details, use_container_width=True, hide_index=True)
        st.json(
            {
                "category": category,
                "matched_skills": matched_skills,
                "feature_columns": FEATURE_COLS,
            }
        )
else:
    st.info("Load the sample or enter your own text, then click Analyze resume.")
