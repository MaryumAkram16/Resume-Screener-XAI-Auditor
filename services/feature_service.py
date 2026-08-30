import re
from difflib import SequenceMatcher

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from .model_service import skill_vectorizer, suitability_vectorizer


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


skill_vocabulary = skill_vectorizer.get_feature_names_out()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


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


def build_features(resume_text, job_text):
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

    return features_df, matched_skills
