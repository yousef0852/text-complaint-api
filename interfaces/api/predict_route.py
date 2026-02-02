"""
HTTP interface for the /predict endpoint.

Responsibilities:
- Define the POST /predict route.
- Accept and validate incoming HTTP requests using schemas.
- Delegate the request to the core pipeline.
- Map domain errors to appropriate HTTP responses.

This module MUST NOT:
- Perform text preprocessing.
- Run ML models or inference.
- Apply business rules or action mapping.
- Contain core decision logic.

This file acts strictly as an HTTP adapter.
"""
from fastapi import APIRouter, HTTPException
from interfaces.schemas.complaint import ComplaintRequest, ComplaintResponse
from core.pipeline import run_pipeline
from fastapi import Depends


from fastapi import APIRouter, Depends, Request, HTTPException

router = APIRouter(prefix="/predict", tags=["Prediction"])

def get_model_loader(request: Request):
    return request.app.state.model_loader

@router.post("", response_model=ComplaintResponse)
async def predict_complaint(request: ComplaintRequest, loader = Depends(get_model_loader)):
    result = run_pipeline(request.text, loader)
    return result

@router.get("/health")
async def health_check():
    return {"status": "ok"}