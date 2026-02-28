from fastapi import APIRouter, Depends, Request, HTTPException
from interfaces.schemas.complaint import ComplaintRequest, ComplaintResponse
from core.pipeline import run_pipeline
from configs.config import settings
import json
import os
from datetime import datetime

router = APIRouter(prefix="/predict", tags=["Prediction"])

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
LOGS_FILE = os.path.join(LOGS_DIR, "predictions.json")

def save_prediction_log(input_text: str, response: ComplaintResponse):
    os.makedirs(LOGS_DIR, exist_ok=True)

    entries = []
    if os.path.exists(LOGS_FILE):
        with open(LOGS_FILE, "r", encoding="utf-8") as f:
            entries = json.load(f)

    entries.append({
        "timestamp": datetime.now().isoformat(),
        "input_text": input_text,
        "response": response.model_dump(mode="json")
    })

    with open(LOGS_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

def get_model_loader(request: Request):
    return request.app.state.model_loader

@router.post("", response_model=ComplaintResponse)
async def predict_complaint(request: ComplaintRequest, loader = Depends(get_model_loader)):
    result = run_pipeline(request.text, loader)
    if settings.ENABLE_PREDICTION_LOGGING:
        save_prediction_log(request.text, result)
    return result