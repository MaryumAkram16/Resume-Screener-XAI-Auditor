# Resume Screener — SHAP Validation Dashboard

An explainable AI dashboard for auditing resume-to-job suitability predictions. This repository extends the Resume Screener model with **SHAP-based feature attribution**, allowing each prediction to be inspected rather than presented as an unexplained score.

The application is implemented with Streamlit and is intended for model validation, prediction auditing, and demonstration of explainable machine learning techniques.

## Live application

https://resume-screener-validation-maryum.streamlit.app/

## What this project validates

The dashboard evaluates an existing resume-screening pipeline at the level of individual predictions. Given a resume and a job description, it displays:

1. A predicted resume category.
2. The probability associated with the predicted category.
3. A suitability score from 0 to 100.
4. The skills shared by the resume and job description.
5. The feature values used by the suitability model.
6. SHAP feature impacts explaining why the suitability score increased or decreased.

> SHAP explanations provide local model explainability. They help audit individual predictions, but they do not replace test-set metrics such as accuracy, MAE, R², or a confusion matrix.

## Why SHAP was added

A single suitability score does not show whether a model is relying on sensible evidence. The SHAP layer makes the prediction more transparent by decomposing the model output into feature-level contributions.

For every analysis, the dashboard shows the features that influenced the prediction most strongly. Positive SHAP values pushed the prediction higher, while negative SHAP values pushed it lower. The dashboard also compares the raw prediction with the SHAP reconstruction to check that the explanation is consistent with the model output.

This makes the application useful as an **XAI auditor** rather than only as a resume-scoring interface.

## Prediction pipeline

The dashboard uses the following workflow:

```text
Resume text + job description
              │
              ▼
     Feature construction
              │
              ├── Skill overlap
              ├── Text similarity
              ├── Skill-text similarity
              ├── Embedding similarity
              ├── Length ratio
              ├── Fuzzy matching
              └── Dataset-provided matching features
              │
              ▼
     Suitability model prediction
              │
              ▼
     SHAP TreeExplainer audit
              │
              ▼
  Score, feature values, and explanations
```

The category classifier runs separately and predicts the most likely resume category with a probability distribution.

## Suitability features

The suitability model uses eight features:

| Feature | Meaning |
|---|---|
| `skill_overlap_ratio` | Proportion of required job skills found in the resume skill vocabulary. |
| `skill_overlap_count` | Number of shared skills between the resume and job description. |
| `length_ratio` | Relative length of the resume and job description text. |
| `text_similarity` | TF-IDF cosine similarity between the complete resume and job text. |
| `skill_text_similarity` | TF-IDF cosine similarity between the extracted skill text. |
| `embedding_similarity` | Sentence-transformer semantic similarity between the resume and job text. |
| `skill_string_match_score` | Skill matching score included in the training data. |
| `fuzzy_match_score` | Fuzzy text matching score included in the training data. |

The feature order is preserved in `services/feature_service.py` and passed to the saved suitability model and SHAP explainer.

## Model explainability output

The dashboard uses `shap.TreeExplainer` for the saved gradient-boosting suitability model. It presents:

- A ranked feature-impact chart.
- The numerical feature values used for the prediction.
- The direction of each feature impact.
- The SHAP base value.
- The reconstructed score from the SHAP explanation.
- The difference between the model score and reconstructed score.

This allows a reviewer to investigate questions such as:

- Did shared skills contribute positively to the score?
- Did the text similarity feature dominate the prediction?
- Which features reduced the predicted suitability?
- Does the explanation reconstruct the model output correctly?

## Repository structure

```text
.
├── dashboard.py                    # SHAP validation and Streamlit interface
├── app.py                          # Original Resume Screener Streamlit interface
├── api.py                          # Optional FastAPI interface
├── services/
│   ├── feature_service.py          # Feature construction and skill matching
│   └── model_service.py            # Model loading, prediction, and SHAP logic
├── category_classifier.pkl         # Saved category classifier
├── tfidf_vectorizer.pkl            # Category TF-IDF vectorizer
├── suitability_model.pkl           # Saved suitability model
├── suitability_vectorizer.pkl      # Suitability TF-IDF vectorizer
├── skill_vectorizer.pkl            # Skill vocabulary vectorizer
├── .streamlit/config.toml          # Streamlit theme and server settings
├── requirements.txt                # Python dependencies
└── runtime.txt                     # Python runtime version
```

## Installation

Use Python 3.11 or a compatible Python environment.

```bash
git clone https://github.com/MaryumAkram16/Resume-Screener-validation.git
cd Resume-Screener-validation
python -m venv .venv
```

Activate the environment on Windows:

```bat
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file must include the package used by the SHAP validation layer:

```text
streamlit==1.59.1
scikit-learn==1.6.1
joblib==1.5.3
numpy==2.0.2
pandas
scipy
sentence-transformers==5.6.0
shap
```

## Run the dashboard locally

Launch the SHAP dashboard with:

```bash
streamlit run dashboard.py
```

The first run may download the `all-MiniLM-L6-v2` sentence-transformer model. The saved `.pkl` artifacts must remain in the repository root because the service modules load them using paths relative to the project directory.

## Deploy with Streamlit Community Cloud

1. Push the repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with GitHub.
4. Select **Create app**.
5. Choose the repository `MaryumAkram16/Resume-Screener-validation`.
6. Select the `main` branch.
7. Set the main file path to `dashboard.py`.
8. Deploy the application.

GitHub Actions or GitHub Pages are not required to host the Streamlit process. Streamlit Community Cloud should read `requirements.txt` and run `dashboard.py` directly.

## Relationship to the original Resume Screener

The original screening application predicts resume categories and suitability scores. This repository focuses on the **validation and explainability layer** added around that prediction pipeline.

The main distinction is:

| Original application | This repository |
|---|---|
| Presents category and suitability predictions. | Audits predictions with feature-level explanations. |
| Focuses on the screening workflow. | Focuses on model transparency and local validation. |
| Displays the final score. | Displays the score, input features, SHAP impacts, and reconstruction details. |

## Limitations

SHAP explains the behavior of the saved model; it does not guarantee that the model is fair, accurate, or suitable for making employment decisions. The explanation is only as reliable as the features, training data, and model itself.

The training data is limited in size and contains approximated or synthetic matching assumptions. The application should therefore be treated as a research and auditing tool, not as an automated hiring decision-maker. Human review remains necessary for any real recruitment process.

The sentence-transformer model may increase startup time and memory usage during deployment. The `.pkl` files must also be generated with compatible versions of Python and scikit-learn.

## Technologies

- [Streamlit](https://streamlit.io/) — interactive dashboard
- [SHAP](https://shap.readthedocs.io/) — model explainability
- [scikit-learn](https://scikit-learn.org/) — TF-IDF, classification, regression, and similarity calculations
- [sentence-transformers](https://www.sbert.net/) — semantic text embeddings
- [joblib](https://joblib.readthedocs.io/) — persisted model and vectorizer loading
- [pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) — data processing

## Responsible use

This project is intended for educational, research, and model-auditing purposes. It should not be used as the sole basis for accepting, rejecting, ranking, or making employment decisions about candidates.
