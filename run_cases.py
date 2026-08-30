import re
from difflib import SequenceMatcher

import joblib
import numpy as np
import pandas as pd
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


def validate_case(example_name, resume_text, job_text):
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

    category = category_model.predict(
        category_vectorizer.transform([resume_text])
    )[0]
    raw_score = suitability_model.predict(features_df)[0]
    display_score = float(np.clip(raw_score, 0, 100))

    print("\n--- " + example_name + " ---")
    print("category:", category)
    print("raw score:", raw_score)
    print("display score:", display_score)
    print("resume skills:", resume_skills)
    print("job skills:", job_skills)
    print("matched skills:", matched_skills)
    print("features:")
    print(features_df.to_string(index=False))
    print("feature shape:", features_df.shape)
    print("feature columns:", list(features_df.columns))

    assert features_df.shape == (1, 8)
    assert list(features_df.columns) == FEATURE_COLS


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


validate_case("PARTIAL MATCH", partial_resume, partial_job)
validate_case("WEAK MATCH", weak_resume, weak_job)

print("\nAll partial-match and weak-match validation checks passed.")
