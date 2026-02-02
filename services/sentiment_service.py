from interfaces.schemas.complaint import PredictionDetail
from interfaces.schemas.enums import SentimentLabel
from transformers import pipeline

def predict_sentiment_service(text: str, model_pipeline: pipeline) -> PredictionDetail:
    results = model_pipeline(text)
    top_result = results[0][0] 
    label_raw = top_result['label']

    mapping = {
        "LABEL_0": SentimentLabel.NEG,
        "LABEL_1": SentimentLabel.NEU,
        "LABEL_2": SentimentLabel.POS
    }
    label_enum = mapping.get(label_raw, SentimentLabel.NEU)

    return PredictionDetail(
        label=label_enum,
        confidence=top_result['score'],
        explanation=f"Sentiment: {label_raw} -> {label_enum.value}",
        low_confidence=(top_result['score'] < 0.7)
    )