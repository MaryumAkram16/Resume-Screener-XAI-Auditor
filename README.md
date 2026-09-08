# SHAP-Based Resume Screening Validation

This project adds an explainability and validation layer to a resume screening system. It uses SHAP (SHapley Additive exPlanations) to show how different resume–job matching features influence the model’s suitability prediction.

## Live Application

- **Streamlit dashboard:** https://resume-screener-validation-maryum.streamlit.app/
- **FastAPI backend (Railway):** https://resume-screener-validation-production.up.railway.app
- **Hand-coded HTML/JS client (GitHub Pages):** https://maryumakram16.github.io/Resume-Screener-validation/

## Architecture

The model and SHAP explanation logic are exposed once, as an API, and consumed by two independent frontends:

```
                    api.py (FastAPI, deployed on Railway)
                    /predict  /explain
                            │
              ┌─────────────┴─────────────┐
              │                           │
      dashboard.py                  index.html
      (Streamlit)                (hand-coded HTML/JS,
                                   Chart.js, no framework)
```

- **`api.py`** loads the trained models once at startup and exposes `/predict` (category + suitability score) and `/explain` (adds the full SHAP breakdown: per-feature impact, direction, and a reconstruction check against the raw model output).
- **`dashboard.py`** is a Streamlit client. It calls the API rather than loading models locally, so it stays lightweight and always reflects whatever is currently deployed on Railway.
- **`index.html`** is a self-contained, hand-coded frontend (no build step, no framework) that calls the same API. It supports both a fixed set of sample resume/job pairs and free-text input, and renders the same SHAP chart and feature table as the Streamlit dashboard.

Because both frontends call the same API, they always stay consistent with each other and with the model itself - there is exactly one place where scoring and explanation logic lives.

## Why SHAP?

A machine-learning model may produce a suitability score, but the score alone does not explain the reason behind the prediction. SHAP helps make the model more transparent by showing the contribution of each feature to an individual prediction.

SHAP is important because it helps to:

- Explain why a resume received a particular suitability score.
- Identify the features that increased or decreased the prediction.
- Check whether the model is relying on meaningful signals.
- Detect unexpected or potentially misleading model behavior.
- Improve trust and interpretability when reviewing predictions.

## How SHAP is used

For each resume and job-description pair, the system first calculates the features used by the suitability model. These include skill overlap, text similarity, semantic similarity, fuzzy matching, and text-length ratio.

The trained model then generates a suitability prediction. SHAP analyzes that prediction and assigns an impact value to each feature:

- A positive SHAP value means the feature increased the predicted suitability score.
- A negative SHAP value means the feature decreased the predicted suitability score.
- A larger absolute SHAP value means the feature had a stronger influence on the prediction.

The system also compares the original model prediction with the score reconstructed from the SHAP values. This helps verify that the explanation is consistent with the model output.

## Features analyzed

| Feature | Purpose |
|---|---|
| `skill_overlap_ratio` | Measures the proportion of job skills found in the resume. |
| `skill_overlap_count` | Counts the shared skills between the resume and job description. |
| `text_similarity` | Measures similarity between resume and job text using TF-IDF. |
| `skill_text_similarity` | Measures similarity between the extracted skill text. |
| `embedding_similarity` | Measures semantic similarity using sentence embeddings. |
| `length_ratio` | Compares the relative lengths of the resume and job description. |
| `skill_string_match_score` | Represents skill-based matching information from the data. |
| `fuzzy_match_score` | Measures approximate text matching between the inputs. |

Note: `skill_string_match_score` and `fuzzy_match_score` are approximations of features that, in the original training data, came from a private matching pipeline that isn't public. The live app computes honest, simplified substitutes for these two rather than trying to reverse-engineer the original.

The list of "skills" a resume/job pair can match against comes from a raw TF-IDF vocabulary fit on the training corpus, not a curated skill list. This means it also contains generic words alongside real skills. The numeric features above are computed from the full vocabulary exactly as they were during training; only the human-facing "matched skills" list is filtered down to remove obvious non-skill terms for display.

## Validation output

For each prediction, the system provides:

- The final suitability score.
- The predicted resume category.
- Matched skills between the resume and job description.
- The values of the features used by the model.
- The SHAP impact of each feature.
- The direction and strength of each feature’s influence.
- A comparison between the original prediction and the SHAP-reconstructed score.

## Repository structure

| File / folder | Purpose |
|---|---|
| `api.py` | FastAPI backend - loads the models once, exposes `/predict` and `/explain` |
| `dashboard.py` | Streamlit UI, calls the API |
| `index.html` | Hand-coded HTML/JS UI, calls the API |
| `services/feature_service.py` | Feature engineering: builds the 8 model inputs from a resume/job pair |
| `services/model_service.py` | Loads the trained models and vectorizers; runs prediction and SHAP explanation |
| `run_cases.py`, `run_cases_shap.py` | Standalone scripts exercising the pipeline against a few hardcoded test cases |
| `*.pkl` | Trained models and fitted vectorizers |
| `requirements.txt` | Shared dependencies for the API and the Streamlit dashboard |

## Importance of the project

This project demonstrates how explainable AI can be applied to resume screening. Instead of treating the model as a black box, it provides evidence for how individual predictions are produced.

The SHAP analysis is intended to support model auditing and interpretation. It does not guarantee that the model is accurate or unbiased, and it should not be used as the sole basis for employment decisions.
