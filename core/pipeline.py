from interfaces.schemas.complaint import ComplaintResponse, PredictionDetail, ActionDetail
from interfaces.schemas.enums import SentimentLabel, TopicLabel, ActionLabel
from utils.text_utils import ArabicInput
from services.model_loader import ModelLoader
from services.sentiment_service import predict_sentiment_service
from services.topic_service import predict_topic_service
from services.action_service import predict_action_service
from configs.config import settings
from configs.logging import get_logger


logger = get_logger("pipeline")

def predict_sentiment(text: str, model_loader: ModelLoader) -> PredictionDetail:
    return predict_sentiment_service(text, model_loader.sentiment_model)


def predict_topic(text: str, model_loader: ModelLoader) -> PredictionDetail:
    return predict_topic_service(text, model_loader.topic_model)


def predict_intent(text: str, model_loader: ModelLoader) -> PredictionDetail:
    return predict_action_service(text, model_loader.action_model)


def map_action(topic: TopicLabel, sentiment: SentimentLabel, action_intent: ActionLabel) -> ActionDetail:
    source = "RULE_ENGINE"
    
    match (topic, sentiment, action_intent):
        case (TopicLabel.POLICY_SECURITY, _, _):
            return ActionDetail(label="BLOCK_AND_REVIEW", decision_source=source)
            
        case (TopicLabel.FINANCIAL, SentimentLabel.NEG, _):
            return ActionDetail(label="FINANCIAL_ESCALATION", decision_source=source)
            
        case (TopicLabel.TECH, _, ActionLabel.REPORT_BUG):
            return ActionDetail(label="CREATE_JIRA_TICKET", decision_source=source)
            
        case (TopicLabel.TECH, SentimentLabel.NEG, _):
            return ActionDetail(label="TECH_SUPPORT_ESCALATION", decision_source=source)
            
        case (TopicLabel.CONTENT, _, ActionLabel.USER_REQUEST):
            return ActionDetail(label="CONTENT_MODIFICATION_QUEUE", decision_source=source)
            
        case (_, SentimentLabel.POS, _):
            return ActionDetail(label="AUTO_REPLY_THANK_YOU", decision_source=source)
            
        case (_, SentimentLabel.NEU, ActionLabel.GENERAL_NOTE):
            return ActionDetail(label="ARCHIVE_NOTE", decision_source=source)
            
        case _:
            return ActionDetail(label="GENERAL_SUPPORT_ROUTING", decision_source=source)


from configs.logging import get_logger
from utils.text_utils import ArabicInput
from interfaces.schemas.complaint import ComplaintResponse

def run_pipeline(text: str, model_loader: ModelLoader) -> ComplaintResponse:
    logger = get_logger("pipeline")

    raw_len = len(text or "")
    # Log text in a way that Railway can handle
    logger.info("pipeline_started", raw_text_len=raw_len)
    print(f"[DEBUG] Input text: {text}")  # Console print as fallback
    logger.info(f"Input text: {text}")

    clean = ArabicInput(text=(text or "").strip())
    logger.info(
        "text_cleaned",
        raw_text_len=raw_len,
        cleaned_text_len=len(clean.text),
    )

    sentiment = predict_sentiment(clean.text, model_loader)
    logger.info(
        "sentiment_predicted",
        label=sentiment.label,
        confidence=sentiment.confidence,
    )

    topic = predict_topic(clean.text, model_loader)
    logger.info(
        "topic_predicted",
        label=topic.label,
        confidence=topic.confidence,
    )

    intent = predict_intent(clean.text, model_loader)
    logger.info(
        "intent_predicted",
        label=intent.label,
        confidence=intent.confidence,
    )

    action = map_action(topic.label, sentiment.label, intent.label)
    logger.info(
        "action_mapped",
        action_label=action.label,
        decision_source=getattr(action, "decision_source", "RULE_ENGINE"),
        rule_inputs={
            "topic": topic.label,
            "sentiment": sentiment.label,
            "intent": intent.label,
        },
    )

    guarding_triggered = False

    if settings.ENABLE_CONFIDENCE_GUARDING:
        s_th = settings.get_threshold("sentiment")
        t_th = settings.get_threshold("topic")
        i_th = settings.get_threshold("intent")

        logger.info(
            "confidence_guard_config",
            enabled=True,
            thresholds={"sentiment": s_th, "topic": t_th, "intent": i_th},
        )

        if sentiment.confidence < s_th:
            guarding_triggered = True
            action.label = "MANUAL_REVIEW"
            action.decision_source = "CONFIDENCE_THRESHOLD"
            logger.warning(
                "confidence_guard_triggered",
                model="sentiment",
                confidence=sentiment.confidence,
                threshold=s_th,
                forced_action=action.label,
            )

        elif topic.confidence < t_th:
            guarding_triggered = True
            action.label = "MANUAL_REVIEW"
            action.decision_source = "CONFIDENCE_THRESHOLD"
            logger.warning(
                "confidence_guard_triggered",
                model="topic",
                confidence=topic.confidence,
                threshold=t_th,
                forced_action=action.label,
            )

        elif intent.confidence < i_th:
            guarding_triggered = True
            action.label = "MANUAL_REVIEW"
            action.decision_source = "CONFIDENCE_THRESHOLD"
            logger.warning(
                "confidence_guard_triggered",
                model="intent",
                confidence=intent.confidence,
                threshold=i_th,
                forced_action=action.label,
            )

    logger.info(
        "pipeline_completed",
        final_action=action.label,
        decision_source=getattr(action, "decision_source", None),
        guarding_triggered=guarding_triggered,
    )

    return ComplaintResponse(
        sentiment=sentiment,
        topic=topic,
        intent=intent,
        action=action,
        meta={
            "model_version": "MARBERT-v2",
            "input_text": text[:100]  # Add input text to response
        },
    )
