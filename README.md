# SHAP-Based Resume Screening Validation

This project adds an explainability and validation layer to a resume screening system. It uses SHAP (SHapley Additive exPlanations) to show how different resume–job matching features influence the model’s suitability prediction.

## Live Application
https://resume-screener-validation-maryum.streamlit.app/


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

## Validation output

For each prediction, the system provides:

- The final suitability score.
- The predicted resume category.
- Matched skills between the resume and job description.
- The values of the features used by the model.
- The SHAP impact of each feature.
- The direction and strength of each feature’s influence.
- A comparison between the original prediction and the SHAP-reconstructed score.

## Importance of the project

This project demonstrates how explainable AI can be applied to resume screening. Instead of treating the model as a black box, it provides evidence for how individual predictions are produced.

The SHAP analysis is intended to support model auditing and interpretation. It does not guarantee that the model is accurate or unbiased, and it should not be used as the sole basis for employment decisions.
