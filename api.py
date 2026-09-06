from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from services.feature_service import FEATURE_COLS, build_features
from services.model_service import (
    explain_suitability,
    predict_category,
    predict_suitability,
)


app = FastAPI(
    title="Resume Screener XAI Auditor API",
    version="1.0.0",
    description="Prediction and SHAP explanations for the validated Resume Screener model.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScreeningRequest(BaseModel):
    resume_text: str = Field(..., min_length=1, description="Resume text")
    job_text: str = Field(..., min_length=1, description="Job description text")

    @field_validator("resume_text", "job_text")
    @classmethod
    def text_must_not_be_blank(cls, value):
        if not value.strip():
            raise ValueError("Text must not be blank")
        return value.strip()


class PredictionResponse(BaseModel):
    category: str
    category_probability: float
    raw_score: float
    display_score: float
    matched_skills: list[str]


class Reason(BaseModel):
    feature: str
    value: float
    impact: float
    direction: str


class ExplanationResponse(PredictionResponse):
    base_value: float
    reconstructed_score: float
    reconstruction_difference: float
    reasons: list[Reason]


def run_screening(request: ScreeningRequest):
    try:
        features_df, matched_skills = build_features(
            request.resume_text,
            request.job_text,
        )

        if features_df.shape != (1, 8):
            raise RuntimeError(
                f"Unexpected feature shape: {features_df.shape}; expected (1, 8)"
            )

        if list(features_df.columns) != FEATURE_COLS:
            raise RuntimeError("Feature columns do not match the model contract")

        category, probabilities = predict_category(request.resume_text)
        raw_score, display_score = predict_suitability(features_df)
        category_probability = float(max(probabilities))

        return {
            "features_df": features_df,
            "matched_skills": matched_skills,
            "category": category,
            "category_probability": category_probability,
            "raw_score": raw_score,
            "display_score": display_score,
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Screening pipeline failed: {error}",
        ) from error


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "Resume Screener XAI Auditor API",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ScreeningRequest):
    result = run_screening(request)

    return PredictionResponse(
        category=result["category"],
        category_probability=result["category_probability"],
        raw_score=result["raw_score"],
        display_score=result["display_score"],
        matched_skills=result["matched_skills"],
    )


@app.post("/explain", response_model=ExplanationResponse)
def explain(request: ScreeningRequest):
    result = run_screening(request)
    shap_result = explain_suitability(
        result["features_df"],
        FEATURE_COLS,
    )

    reconstructed_score = shap_result["reconstructed_score"]
    reconstruction_difference = abs(
        result["raw_score"] - reconstructed_score
    )

    return ExplanationResponse(
        category=result["category"],
        category_probability=result["category_probability"],
        raw_score=result["raw_score"],
        display_score=result["display_score"],
        matched_skills=result["matched_skills"],
        base_value=shap_result["base_value"],
        reconstructed_score=reconstructed_score,
        reconstruction_difference=reconstruction_difference,
        reasons=shap_result["reasons"],
    )
