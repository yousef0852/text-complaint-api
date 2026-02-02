"""
Core pipeline orchestration module.

Responsibilities:
- Define the end-to-end prediction workflow.
- Orchestrate pipeline steps in the correct order:
  validation -> preprocessing -> classifiers -> action mapping.
- Coordinate calls to underlying services.
- Return a domain-level result independent of HTTP concerns.

This module MUST NOT:
- Depend on FastAPI or HTTP concepts.
- Handle request/response serialization.
- Know anything about routes, status codes, or headers.

This file represents the main application use case.
"""

from interfaces.schemas.complaint import ComplaintResponse, PredictionDetail, ActionDetail
from interfaces.schemas.enums import SentimentLabel, TopicLabel, ActionLabel
from utils.text_utils import ArabicInput
from services.model_loader import ModelLoader
from services.sentiment_service import predict_sentiment_service
from services.topic_service import predict_topic_service
from services.action_service import predict_action_service

def predict_sentiment(text: str, model_loader: ModelLoader) -> PredictionDetail:
    # TODO: Implement sentiment prediction
    return predict_sentiment_service(text, model_loader.sentiment_model)


def predict_topic(text: str, model_loader: ModelLoader) -> PredictionDetail:
    # TODO: Implement topic prediction
    return predict_topic_service(text, model_loader.topic_model)


def predict_intent(text: str, model_loader: ModelLoader) -> PredictionDetail:
    # TODO: Implement intent prediction
    return predict_action_service(text, model_loader.action_model)


def map_action(topic: TopicLabel, sentiment: SentimentLabel, action_intent: ActionLabel) -> ActionDetail:
    source = "RULE_ENGINE"
    
    match (topic, sentiment, action_intent):
        case (TopicLabel.POLICY_SECURITY, _, _):
            return ActionDetail(label="BLOCK_AND_REVIEW", decision_source=source)
            
        # Financial complaints with negative sentiment
        case (TopicLabel.FINANCIAL, SentimentLabel.NEG, _):
            return ActionDetail(label="FINANCIAL_ESCALATION", decision_source=source)
            
        # Tech bug reports
        case (TopicLabel.TECH, _, ActionLabel.REPORT_BUG):
            return ActionDetail(label="CREATE_JIRA_TICKET", decision_source=source)
            
        # Negative tech support
        case (TopicLabel.TECH, SentimentLabel.NEG, _):
            return ActionDetail(label="TECH_SUPPORT_ESCALATION", decision_source=source)
            
        # Content modification requests
        case (TopicLabel.CONTENT, _, ActionLabel.USER_REQUEST):
            return ActionDetail(label="CONTENT_MODIFICATION_QUEUE", decision_source=source)
            
        # Positive feedback
        case (_, SentimentLabel.POS, _):
            return ActionDetail(label="AUTO_REPLY_THANK_YOU", decision_source=source)
            
        # Neutral general notes
        case (_, SentimentLabel.NEU, ActionLabel.GENERAL_NOTE):
            return ActionDetail(label="ARCHIVE_NOTE", decision_source=source)
            
        # Default case
        case _:
            return ActionDetail(label="GENERAL_SUPPORT_ROUTING", decision_source=source)


def run_pipeline(text: str, model_loader: ModelLoader) -> ComplaintResponse:
  # Strip whitespace from input text
  clean_text = ArabicInput(text=text.strip())
  sentiment = predict_sentiment(clean_text.text, model_loader)
  topic = predict_topic(clean_text.text, model_loader)
  intent = predict_intent(clean_text.text, model_loader)
  action = map_action(topic.label, sentiment.label, intent.label)
  if sentiment.confidence < 0.7 or topic.confidence < 0.7 or intent.confidence < 0.7:
    action.label = "MANUAL_REVIEW"
    action.decision_source = "CONFIDENCE_THRESHOLD"
  return ComplaintResponse(
    sentiment=sentiment,
    topic=topic,
    intent=intent,
    action=action,
    meta={"model_version": "MARBERT-v2"}
  )
