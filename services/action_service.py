from interfaces.schemas.complaint import PredictionDetail
from interfaces.schemas.enums import ActionLabel
from transformers import pipeline

def predict_action_service(text: str, model_pipeline: pipeline) -> PredictionDetail:
    results = model_pipeline(text)
    top_result = results[0][0]
    label_raw = top_result['label']

    mapping = {
        "LABEL_0": ActionLabel.GENERAL_NOTE,
        "LABEL_1": ActionLabel.USER_REQUEST,
        "LABEL_2": ActionLabel.REPORT_BUG
    }
    label_enum = mapping.get(label_raw, ActionLabel.USER_REQUEST)

    return PredictionDetail(
        label=label_enum,
        confidence=top_result['score'],
        explanation=f"Action: {label_raw} -> {label_enum.value}",
        low_confidence=(top_result['score'] < 0.7)
    )