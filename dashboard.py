import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="Resume Screener XAI Auditor",
    page_icon="📄",
    layout="wide",
)

API_BASE = "https://resume-screener-validation-production.up.railway.app"


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

    with st.spinner("Calculating prediction and SHAP explanation... (may take a moment if the API is waking up)"):
        try:
            response = requests.post(
                f"{API_BASE}/explain",
                json={"resume_text": resume_text.strip(), "job_text": job_text.strip()},
                timeout=90,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as error:
            st.error(f"The screening pipeline failed: {error}")
            st.stop()

    reasons = pd.DataFrame(data["reasons"])
    reasons["absolute_impact"] = reasons["impact"].abs()
    reasons = reasons.sort_values("absolute_impact", ascending=False)

    st.success("Analysis completed successfully.")

    score_col, category_col, match_col = st.columns(3)
    with score_col:
        st.metric("Suitability score", f"{data['display_score']:.2f} / 100")
    with category_col:
        st.metric("Predicted category", data["category"])
        st.caption(f"Category probability: {data['category_probability']:.2%}")
    with match_col:
        st.metric("Matched skills", len(data["matched_skills"]))
        st.caption(", ".join(data["matched_skills"]) if data["matched_skills"] else "No shared vocabulary skills")

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
        st.dataframe(reasons[["feature", "value"]], use_container_width=True, hide_index=True)
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
                    data["raw_score"],
                    data["display_score"],
                    data["base_value"],
                    data["reconstructed_score"],
                    abs(data["raw_score"] - data["reconstructed_score"]),
                ],
            }
        )
        st.dataframe(details, use_container_width=True, hide_index=True)
        st.json(
            {
                "category": data["category"],
                "matched_skills": data["matched_skills"],
                "feature_columns": list(reasons["feature"]),
            }
        )
else:
    st.info("Load the sample or enter your own text, then click Analyze resume.")
