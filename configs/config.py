import os
from typing import Optional

class Settings:
    # Hugging Face
    HF_TOKEN: str = os.getenv("HF_TOKEN")
    
    # Model Thresholds
    SENTIMENT_THRESHOLD: float = float(os.getenv("SENTIMENT_THRESHOLD", "0.7"))
    TOPIC_THRESHOLD: float = float(os.getenv("TOPIC_THRESHOLD", "0.7"))
    INTENT_THRESHOLD: float = float(os.getenv("INTENT_THRESHOLD", "0.7"))

    ENABLE_CONFIDENCE_GUARDING: bool = os.getenv("ENABLE_CONFIDENCE_GUARDING", "true").lower() == "true"
    MANUAL_REVIEW_ON_LOW_CONFIDENCE: bool = os.getenv("MANUAL_REVIEW_ON_LOW_CONFIDENCE", "true").lower() == "true"
    ENABLE_PREDICTION_LOGGING: bool = os.getenv("ENABLE_PREDICTION_LOGGING", "false").lower() == "true"

    @classmethod
    def get_threshold(cls, model_type: str) -> float:
        thresholds = {
            "sentiment": cls.SENTIMENT_THRESHOLD,
            "topic": cls.TOPIC_THRESHOLD,
            "intent": cls.INTENT_THRESHOLD
        }
        return thresholds.get(model_type, 0.7)
 
settings = Settings()
    