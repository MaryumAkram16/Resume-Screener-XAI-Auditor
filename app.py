import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Resume Screener",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE = "https://resume-screener-validation-production.up.railway.app"


def score_resume_against_job(resume_text, job_text):
    try:
        response = requests.post(
            f"{API_BASE}/explain",
            json={"resume_text": resume_text, "job_text": job_text},
            timeout=90,
        )
    except requests.exceptions.RequestException as error:
        st.error(f"Could not reach the scoring API: {error}")
        st.stop()

    if response.status_code != 200:
        st.error(f"Scoring API returned an error: {response.text}")
        st.stop()

    data = response.json()

    classes = list(data["category_probabilities"].keys())
    proba = list(data["category_probabilities"].values())
    order = sorted(range(len(proba)), key=lambda i: proba[i], reverse=True)

    return {
        "category": data["category"],
        "category_confidence": proba,
        "classes": classes,
        "order": order,
        "suitability_score": data["display_score"],
        "raw_score": data["raw_score"],
        "base_value": data["base_value"],
        "reconstructed_score": data["reconstructed_score"],
        "matched_skills": data["matched_skills"],
        "reasons": data["reasons"],
    }


# ============ GLOBAL DARK THEME STYLING ============
st.markdown("""
<style>
:root {
    --bg-main: #0A0E1A;
    --bg-card: #131829;
    --bg-card-alt: #1A2036;
    --border: #262D45;
    --purple: #8B5CF6;
    --purple-dark: #6D28D9;
    --teal: #2DD4BF;
    --text-main: #E8EAF0;
    --text-muted: #8891A8;
}

.stApp {
    background-color: var(--bg-main);
    color: var(--text-main);
}

section[data-testid="stSidebar"] {
    background-color: #0D1220;
    border-right: 1px solid var(--border);
}

h1, h2, h3, h4, h5, p, span, div, label {
    color: var(--text-main);
}

.eyebrow {
    display: inline-block;
    color: var(--teal);
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    font-weight: 600;
    text-transform: uppercase;
    background: rgba(45, 212, 191, 0.1);
    border: 1px solid rgba(45, 212, 191, 0.3);
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}

.hero {
    padding: 1rem 0 2rem 0;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 1rem;
    color: #FFFFFF;
}
.hero p {
    color: var(--text-muted);
    font-size: 1.05rem;
    max-width: 640px;
    line-height: 1.6;
}

.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.badge {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    margin-right: 0.4rem;
}
.badge-purple { background: rgba(139, 92, 246, 0.15); color: #C4B5FD; border: 1px solid rgba(139, 92, 246, 0.3); }
.badge-teal { background: rgba(45, 212, 191, 0.15); color: #5EEAD4; border: 1px solid rgba(45, 212, 191, 0.3); }
.badge-warn { background: rgba(245, 158, 11, 0.15); color: #FCD34D; border: 1px solid rgba(245, 158, 11, 0.3); }

.result-card {
    background: linear-gradient(135deg, rgba(139,92,246,0.12) 0%, rgba(45,212,191,0.08) 100%);
    border: 1px solid rgba(139, 92, 246, 0.35);
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin-top: 1rem;
}
.result-label { font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }
.result-value { font-size: 1.8rem; font-weight: 800; color: #FFFFFF; margin: 0.3rem 0; }

.conf-row { margin-bottom: 0.9rem; }
.conf-label { display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 0.3rem; }
.conf-track { background: #1E2438; border-radius: 8px; height: 10px; overflow: hidden; }
.conf-fill { background: linear-gradient(90deg, var(--purple), var(--teal)); height: 100%; border-radius: 8px; }

.stButton > button {
    background: linear-gradient(135deg, var(--purple), var(--purple-dark));
    color: white;
    border: none;
    font-weight: 700;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #9D6FFF, var(--purple));
    color: white;
}

.stTextArea textarea, .stTextInput input {
    background: var(--bg-card-alt) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

[data-testid="stMetricValue"] { color: #FFFFFF; }
[data-testid="stMetricLabel"] { color: var(--text-muted); }

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("## 🧾 Resume AI")
    st.caption("Category + suitability screening")
    st.markdown("---")
    st.caption("Student project · Resume + job-posting datasets\nNot an official hiring tool")

# ============ TRY IT ============
st.markdown('<div class="eyebrow">● TWO-STAGE PIPELINE: CATEGORY + SUITABILITY</div>', unsafe_allow_html=True)
st.markdown("""
<div class="hero">
    <h1>Paste a resume and a job.<br>See where it lands.</h1>
    <p>Stage 1 predicts the resume's job category. Stage 2 scores how well it
    actually fits the specific job description you paste in, using a model
    trained on real resume-job match data.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    resume_text = st.text_area(
        "Resume text", height=220,
        placeholder="Paste the resume text here..."
    )
with col2:
    job_text = st.text_area(
        "Job description", height=220,
        placeholder="Paste the job description here..."
    )
predict_clicked = st.button("🔎  Score This Resume", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

if predict_clicked:
    if resume_text.strip() == "" or job_text.strip() == "":
        st.warning("Paste both a resume and a job description first.")
    else:
        with st.spinner("Scoring... (may take a moment if the API is waking up)"):
            result = score_resume_against_job(resume_text, job_text)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Predicted Category</div>
                <div class="result-value">{result['category']}</div>
                <span class="badge badge-purple">Stage 1 · Logistic Regression</span>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            score = result["suitability_score"]
            badge_class = "badge-teal" if score >= 60 else "badge-warn"
            fit_label = "Strong fit" if score >= 60 else ("Moderate fit" if score >= 30 else "Weak fit")
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Suitability Score</div>
                <div class="result-value">{score:.1f} / 100</div>
                <span class="badge {badge_class}">{fit_label}</span>
                <span class="badge badge-purple">Stage 2 · Gradient Boosting</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Category confidence breakdown")
        classes, order, proba = result["classes"], result["order"], result["category_confidence"]
        bars_html = '<div class="card">'
        for i in order[:5]:
            pct = proba[i] * 100
            bars_html += (
                '<div class="conf-row">'
                f'<div class="conf-label"><span>{classes[i]}</span><span>{pct:.1f}%</span></div>'
                f'<div class="conf-track"><div class="conf-fill" style="width:{pct}%;"></div></div>'
                '</div>'
            )
        bars_html += '</div>'
        st.markdown(bars_html, unsafe_allow_html=True)

        st.markdown("#### Matched skills")
        if result["matched_skills"]:
            skill_badges = "".join(
                f'<span class="badge badge-teal">{s}</span>' for s in result["matched_skills"]
            )
            st.markdown(f'<div class="card">{skill_badges}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card">No overlapping skills detected between the resume and job description.</div>', unsafe_allow_html=True)

        reasons = pd.DataFrame(result["reasons"])
        reasons["absolute_impact"] = reasons["impact"].abs()
        reasons_sorted = reasons.sort_values("absolute_impact", ascending=False)

        tab_explanation, tab_features, tab_details = st.tabs(
            ["Why this score?", "Feature values", "Technical details"]
        )

        with tab_explanation:
            st.write(
                "Positive values pushed the model score higher. Negative values pushed it lower."
            )
            chart_data = reasons_sorted.set_index("feature")["impact"]
            st.bar_chart(chart_data, horizontal=True)

            display_reasons = reasons_sorted[["feature", "value", "impact", "direction"]].copy()
            display_reasons["value"] = display_reasons["value"].map(lambda v: f"{v:.6f}")
            display_reasons["impact"] = display_reasons["impact"].map(lambda v: f"{v:+.6f}")
            st.dataframe(display_reasons, use_container_width=True, hide_index=True)

        with tab_features:
            st.subheader("Validated model features")
            st.dataframe(reasons[["feature", "value"]], use_container_width=True, hide_index=True)
            st.caption("The feature order is preserved exactly as expected by the saved suitability model.")

        with tab_details:
            st.subheader("Prediction details")
            details = pd.DataFrame({
                "item": [
                    "Raw suitability score",
                    "Display suitability score",
                    "SHAP base value",
                    "Reconstructed score",
                    "Reconstruction difference",
                ],
                "value": [
                    result["raw_score"],
                    result["suitability_score"],
                    result["base_value"],
                    result["reconstructed_score"],
                    abs(result["raw_score"] - result["reconstructed_score"]),
                ],
            })
            st.dataframe(details, use_container_width=True, hide_index=True)
            st.json({
                "category": result["category"],
                "matched_skills": result["matched_skills"],
                "feature_columns": list(reasons["feature"]),
            })

        st.caption(
            "Suitability score is a trained model estimate, not a hiring decision. "
            "Two of its inputs (skill_string_match_score, fuzzy_match_score) are this app's "
            "own approximations."
        )
