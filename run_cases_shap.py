import re
from difflib import SequenceMatcher

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# Load the saved artifacts from the current project folder.
category_model = joblib.load("category_classifier.pkl")
suitability_model = joblib.load("suitability_model.pkl")
category_vectorizer = joblib.load("tfidf_vectorizer.pkl")
suitability_vectorizer = joblib.load("suitability_vectorizer.pkl")
skill_vectorizer = joblib.load("skill_vectorizer.pkl")


FEATURE_COLS = [
    "skill_overlap_ratio",
    "skill_overlap_count",
    "length_ratio",
    "text_similarity",
    "skill_text_similarity",
    "embedding_similarity",
    "skill_string_match_score",
    "fuzzy_match_score",
]


print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating SHAP TreeExplainer...")
explainer = shap.TreeExplainer(suitability_model)

skill_vocabulary = skill_vectorizer.get_feature_names_out()


def extract_skills(text):
    text_lower = text.lower()
    return [
        skill
        for skill in skill_vocabulary
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower)
    ]


def skill_string_match_stub(resume_text, job_skills):
    if not job_skills:
        return 0.0

    text_lower = resume_text.lower()
    matches = sum(
        1
        for skill in job_skills
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower)
    )
    return 100 * matches / len(job_skills)


def fuzzy_match_stub(resume_skills, job_skills):
    resume_string = " ".join(resume_skills)
    job_string = " ".join(job_skills)

    if not resume_string or not job_string:
        return 0.0

    return SequenceMatcher(None, resume_string, job_string).ratio() * 100


def calculate_features(resume_text, job_text):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)

    resume_set = set(resume_skills)
    job_set = set(job_skills)
    matched_skills = sorted(resume_set & job_set)

    skill_overlap_ratio = (
        len(resume_set & job_set) / len(job_set)
        if job_set
        else 0.0
    )
    skill_overlap_count = len(resume_set & job_set)

    resume_word_count = len(resume_text.split())
    job_word_count = len(job_text.split())
    length_ratio = min(resume_word_count, job_word_count) / max(
        resume_word_count, job_word_count, 1
    )

    text_similarity = cosine_similarity(
        suitability_vectorizer.transform([resume_text]),
        suitability_vectorizer.transform([job_text]),
    )[0][0]

    skill_text_similarity = cosine_similarity(
        skill_vectorizer.transform([" ".join(resume_skills)]),
        skill_vectorizer.transform([" ".join(job_skills)]),
    )[0][0]

    resume_embedding = embed_model.encode([resume_text])
    job_embedding = embed_model.encode([job_text])
    embedding_similarity = cosine_similarity(
        resume_embedding,
        job_embedding,
    )[0][0]

    features_df = pd.DataFrame(
        [
            [
                skill_overlap_ratio,
                skill_overlap_count,
                length_ratio,
                text_similarity,
                skill_text_similarity,
                embedding_similarity,
                skill_string_match_stub(resume_text, job_skills),
                fuzzy_match_stub(resume_skills, job_skills),
            ]
        ],
        columns=FEATURE_COLS,
    )

    return features_df, resume_skills, job_skills, matched_skills


def explain_case(example_name, resume_text, job_text):
    features_df, resume_skills, job_skills, matched_skills = calculate_features(
        resume_text,
        job_text,
    )

    category = category_model.predict(
        category_vectorizer.transform([resume_text])
    )[0]
    raw_score = float(suitability_model.predict(features_df)[0])
    display_score = float(np.clip(raw_score, 0, 100))

    shap_values = explainer.shap_values(features_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_row = np.asarray(shap_values)[0]
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    reconstructed_score = float(base_value + shap_row.sum())
    reconstruction_difference = abs(raw_score - reconstructed_score)

    print("\n--- " + example_name + " ---")
    print("category:", category)
    print("raw score:", raw_score)
    print("display score:", display_score)
    print("resume skills:", resume_skills)
    print("job skills:", job_skills)
    print("matched skills:", matched_skills)
    print("feature shape:", features_df.shape)
    print("feature columns:", list(features_df.columns))
    print("base value:", base_value)
    print("SHAP explanation:")

    explanation_rows = []
    for feature_name, feature_value, impact in zip(
        FEATURE_COLS,
        features_df.iloc[0].values,
        shap_row,
    ):
        direction = "increases" if impact > 0 else "decreases"
        print(
            f"  {feature_name}: value={feature_value:.6f}, "
            f"impact={impact:.6f}, {direction} the score"
        )
        explanation_rows.append(
            {
                "feature": feature_name,
                "feature_value": float(feature_value),
                "shap_impact": float(impact),
                "direction": direction,
            }
        )

    print("reconstructed score:", reconstructed_score)
    print("reconstruction difference:", reconstruction_difference)

    assert features_df.shape == (1, 8)
    assert list(features_df.columns) == FEATURE_COLS
    assert reconstruction_difference < 1e-4

    return {
        "example": example_name,
        "category": category,
        "raw_score": raw_score,
        "display_score": display_score,
        "matched_skills": matched_skills,
        "explanations": explanation_rows,
    }


strong_resume = """
Software engineer with experience in Python, SQL, machine learning,
and backend application development. Worked on data pipelines and APIs.
""".strip()

strong_job = """
We are seeking a backend engineer with Python, SQL, REST APIs,
and machine learning experience.
""".strip()

partial_resume = """
Data analyst experienced in Python, SQL, Excel, dashboards,
and business reporting. Worked with data cleaning and visualization.
""".strip()

partial_job = """
We need a machine learning engineer with Python, SQL, REST APIs,
TensorFlow, model deployment, and cloud experience.
""".strip()

weak_resume = """
Graphic designer experienced in Adobe Photoshop, Illustrator,
branding, typography, visual design, and print materials.
""".strip()

weak_job = """
We are seeking a backend software engineer with Python, SQL,
REST APIs, databases, machine learning, and cloud deployment experience.
""".strip()


results = []
results.append(explain_case("STRONG MATCH", strong_resume, strong_job))
results.append(explain_case("PARTIAL MATCH", partial_resume, partial_job))
results.append(explain_case("WEAK MATCH", weak_resume, weak_job))

print("\n--- SUMMARY ---")
for result in results:
    print(
        f"{result['example']}: "
        f"category={result['category']}, "
        f"display_score={result['display_score']:.6f}, "
        f"matched_skills={len(result['matched_skills'])}"
    )

print("\nAll validation and SHAP reconstruction checks passed.")
