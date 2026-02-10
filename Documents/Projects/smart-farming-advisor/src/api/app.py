"""
FastAPI backend for Smart Farming Advisor
Exposes crop recommendation, disease detection, and farming Q&A endpoints.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.tools.crop_predictor_tool import CropPredictorTool
from src.tools.disease_detector_tool import DiseaseDetectorTool
from src.tools.rag_qa_tool import RAGQATool

app = FastAPI(title="Smart Farming Advisor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

crop_tool = CropPredictorTool()
disease_tool = DiseaseDetectorTool()
qa_tool = RAGQATool()


class CropRequest(BaseModel):
    N: float = Field(..., ge=0)
    P: float = Field(..., ge=0)
    K: float = Field(..., ge=0)
    temperature: float
    humidity: float = Field(..., ge=0, le=100)
    ph: float = Field(..., ge=0, le=14)
    rainfall: float = Field(..., ge=0)


class CropTopRecommendation(BaseModel):
    crop: str
    probability: float


class CropResponse(BaseModel):
    recommended_crop: str
    confidence: float
    top_3_recommendations: List[CropTopRecommendation]
    input_conditions: dict


class QARequest(BaseModel):
    question: str = Field(..., min_length=3)
    use_llm: bool = False


class QADoc(BaseModel):
    content: str
    distance: float
    relevance_score: float


class QAResponse(BaseModel):
    question: str
    answer: str
    source: str
    retrieved_docs: List[QADoc]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/crop/recommend", response_model=CropResponse)
def recommend_crop(payload: CropRequest):
    result = crop_tool.predict(
        N=payload.N,
        P=payload.P,
        K=payload.K,
        temperature=payload.temperature,
        humidity=payload.humidity,
        ph=payload.ph,
        rainfall=payload.rainfall,
        verbose=False,
    )

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@app.post("/api/disease/predict")
async def predict_disease(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    result = disease_tool.predict(str(tmp_path), verbose=False)

    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@app.post("/api/qa", response_model=QAResponse)
def answer_question(payload: QARequest):
    result = qa_tool.answer_question(payload.question, use_llm=payload.use_llm, verbose=False)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result
