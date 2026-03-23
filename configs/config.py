from dotenv import load_dotenv

load_dotenv()

import os
from typing import Optional

class Settings:
    # Hugging Face
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

    # OpenAI-compatible LLM (explanation layer only; does not change routing)
    LLM_ENABLED: bool = os.getenv("LLM_ENABLED", "true").lower() == "true"
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4.1")
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    LLM_MAX_COMPLETION_TOKENS: int = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "512"))
    
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

    def llm_configured(self) -> bool:
        key = self.OPENAI_API_KEY
        return bool(key and str(key).strip())

settings = Settings()
    