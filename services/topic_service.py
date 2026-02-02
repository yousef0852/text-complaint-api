from interfaces.schemas.complaint import PredictionDetail
from interfaces.schemas.enums import TopicLabel
from transformers import pipeline

def predict_topic_service(text: str, model_pipeline: pipeline) -> PredictionDetail:
    results = model_pipeline(text)
    top_result = results[0][0]
    label_raw = top_result['label']

    mapping = {
        "LABEL_0": TopicLabel.POLICY_SECURITY,
        "LABEL_1": TopicLabel.FINANCIAL,
        "LABEL_2": TopicLabel.TECH,
        "LABEL_3": TopicLabel.CONTENT
    }
    label_enum = mapping.get(label_raw, TopicLabel.TECH)

    return PredictionDetail(
        label=label_enum,
        confidence=top_result['score'],
        explanation=f"Topic: {label_raw} -> {label_enum.value}",
        low_confidence=(top_result['score'] < 0.7)
    )