from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from interfaces.api.predict_route import router as predict_router
from interfaces.api.explain_route import router as explain_router
from contextlib import asynccontextmanager
from services.model_loader import ModelLoader
from configs.exceptions import ModelLoadError, ConfigurationError, PredictionError
import os
from configs.logging import setup_logging, get_logger
from interfaces.api.middlewares import RequestIdMiddleware





@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_loader = None
    print(" Environment Variables Check:")
    hf_token = os.getenv("HF_TOKEN")
    print(f"   HF_TOKEN: {'set' if hf_token else 'missing'}")
    if not hf_token:
        print("   HF_TOKEN not found - models might fail to load!")
    
    print("Loading model...")
    try:
        loader = ModelLoader()
        loader.load_models()
        app.state.model_loader = loader
        print(" All models loaded successfully!")
    except ModelLoadError as e:
        print(f" Model loading failed: {e}")
        print(f"   Model: {e.model_name}")
        print(f"   Reason: {e.reason}")
        print(f"   Error Code: {e.error_code}")
        print(" Continuing without models for debugging...")
    except ConfigurationError as e:
        print(f" Configuration error: {e}")
        print(f"   Config Key: {e.config_key}")
        print(f"   Reason: {e.reason}")
        print(f"   Error Code: {e.error_code}")
        print(" Continuing without models for debugging...")
    except Exception as e:
        print(f" Unexpected error during model loading: {e}")
        print(" Continuing without models for debugging...")
    
    yield
    
    if hasattr(app.state, 'model_loader'):
        app.state.model_loader = None
    print("Model unloaded")

setup_logging()
app = FastAPI(lifespan=lifespan)
app.add_middleware(RequestIdMiddleware)
logger = get_logger("errors")

@app.exception_handler(ModelLoadError)
async def model_load_exception_handler(request: Request, exc: ModelLoadError):
    logger.error("Model loading failed", exc_info=True)
    return JSONResponse(
        status_code=503,
        content={
            "error": "MODEL_LOAD_ERROR",
            "message": f"Failed to load model {exc.model_name}: {exc.reason}",
            "error_code": exc.error_code,
            "details": {
                "model_name": exc.model_name,
                "reason": exc.reason
            }
        }
    )

@app.exception_handler(ConfigurationError)
async def config_exception_handler(request: Request, exc: ConfigurationError):
    logger.error("Configuration error", exc_info=True)
    return JSONResponse(
        status_code=400,
        content={
            "error": "CONFIGURATION_ERROR",
            "message": f"Configuration error for {exc.config_key}: {exc.reason}",
            "error_code": exc.error_code,
            "details": {
                "config_key": exc.config_key,
                "reason": exc.reason
            }
        }
    )

@app.exception_handler(PredictionError)
async def prediction_exception_handler(request: Request, exc: PredictionError):
    logger.error("Prediction error", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "PREDICTION_ERROR",
            "message": f"Prediction failed: {exc.reason}",
            "error_code": exc.error_code,
            "details": {
                "text": exc.text[:100] if exc.text else "",
                "reason": exc.reason
            }
        }
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(explain_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/debug/env")
def debug_env():
    from configs.config import settings
    
    return {
        "HF_TOKEN": {
            "exists": settings.HF_TOKEN is not None,
            "length": len(settings.HF_TOKEN) if settings.HF_TOKEN else 0,
        },
        "thresholds": {
            "sentiment": settings.SENTIMENT_THRESHOLD,
            "topic": settings.TOPIC_THRESHOLD,
            "intent": settings.INTENT_THRESHOLD
        },
        "flags": {
            "confidence_guarding": settings.ENABLE_CONFIDENCE_GUARDING,
            "manual_review": settings.MANUAL_REVIEW_ON_LOW_CONFIDENCE
        }
    }