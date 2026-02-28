import pytest
from unittest.mock import patch, MagicMock
from core.pipeline import (
    predict_sentiment,
    predict_topic,
    predict_intent,
    map_action,
    run_pipeline
)
from interfaces.schemas.complaint import PredictionDetail, ActionDetail, ComplaintResponse
from interfaces.schemas.enums import SentimentLabel, TopicLabel, ActionLabel
from services.model_loader import ModelLoader

class MockModelLoader:
    def __init__(self):
        self.sentiment_model = MagicMock(return_value=[[{"label":"LABEL_0","score":0.95}]])
        self.topic_model = MagicMock(return_value=[[{"label":"LABEL_2","score":0.90}]])
        self.action_model = MagicMock(return_value=[[{"label":"LABEL_1","score":0.85}]])
        self.device = -1

class TestPredictSentiment:
    def test_predict_sentiment_returns_prediction_detail(self):
        mock_loader = MockModelLoader()
        result = predict_sentiment("This is a test", mock_loader)
        assert isinstance(result, PredictionDetail)
        assert hasattr(result, 'label')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'explanation')

class TestPredictTopic:
    def test_predict_topic_returns_prediction_detail(self):
        mock_loader = MockModelLoader()
        result = predict_topic("This is a test", mock_loader)
        assert isinstance(result, PredictionDetail)
        assert hasattr(result, 'label')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'explanation')

class TestPredictIntent:
    def test_predict_intent_returns_prediction_detail(self):
        mock_loader = MockModelLoader()
        result = predict_intent("This is a test", mock_loader)
        assert isinstance(result, PredictionDetail)
        assert hasattr(result, 'label')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'explanation')

class TestMapAction:
    def test_map_action_security_topic_blocks(self):
        result = map_action(
            sentiment=SentimentLabel.POS,
            topic=TopicLabel.POLICY_SECURITY,
            action_intent=ActionLabel.USER_REQUEST
        )
        assert result.label == "BLOCK_AND_REVIEW"
        assert result.decision_source == "RULE_ENGINE"

    def test_map_action_negative_financial_escalates(self):
        result = map_action(
            sentiment=SentimentLabel.NEG,
            topic=TopicLabel.FINANCIAL,
            action_intent=ActionLabel.USER_REQUEST
        )
        assert result.label == "FINANCIAL_ESCALATION"

    def test_map_action_tech_bug_creates_ticket(self):
        result = map_action(
            sentiment=SentimentLabel.NEU,
            topic=TopicLabel.TECH,
            action_intent=ActionLabel.REPORT_BUG
        )
        assert result.label == "CREATE_JIRA_TICKET"

    def test_map_action_negative_tech_escalates(self):
        result = map_action(
            sentiment=SentimentLabel.NEG,
            topic=TopicLabel.TECH,
            action_intent=ActionLabel.USER_REQUEST
        )
        assert result.label == "TECH_SUPPORT_ESCALATION"

    def test_map_action_content_modification(self):
        result = map_action(
            sentiment=SentimentLabel.NEU,
            topic=TopicLabel.CONTENT,
            action_intent=ActionLabel.USER_REQUEST
        )
        assert result.label == "CONTENT_MODIFICATION_QUEUE"

    def test_map_action_positive_sentiment_thanks(self):
        result = map_action(
            sentiment=SentimentLabel.POS,
            topic=TopicLabel.CONTENT,
            action_intent=ActionLabel.GENERAL_NOTE
        )
        assert result.label == "AUTO_REPLY_THANK_YOU"

    def test_map_action_neutral_note_archives(self):
        result = map_action(
            sentiment=SentimentLabel.NEU,
            topic=TopicLabel.CONTENT,
            action_intent=ActionLabel.GENERAL_NOTE
        )
        assert result.label == "ARCHIVE_NOTE"

    def test_map_action_default_routing(self):
        result = map_action(
            sentiment=SentimentLabel.NEU,
            topic=TopicLabel.FINANCIAL,
            action_intent=ActionLabel.USER_REQUEST
        )
        assert result.label == "GENERAL_SUPPORT_ROUTING"

class TestRunPipeline:
    @patch('core.pipeline.predict_sentiment')
    @patch('core.pipeline.predict_topic')
    @patch('core.pipeline.predict_intent')
    def test_run_pipeline_returns_complaint_response(self, mock_intent, mock_topic, mock_sentiment):
        mock_loader = MockModelLoader()
        mock_sentiment.return_value = PredictionDetail(
            label=SentimentLabel.NEG,
            confidence=0.95,
            explanation="Test sentiment"
        )
        mock_topic.return_value = PredictionDetail(
            label=TopicLabel.TECH,
            confidence=0.9,
            explanation="Test topic"
        )
        mock_intent.return_value = PredictionDetail(
            label=ActionLabel.USER_REQUEST,
            confidence=0.85,
            explanation="Test intent"
        )

        result = run_pipeline("This is a test complaint", mock_loader)

        assert isinstance(result, ComplaintResponse)
        assert hasattr(result, 'sentiment')
        assert hasattr(result, 'topic')
        assert hasattr(result, 'intent')
        assert hasattr(result, 'action')

        assert result.action.label == "TECH_SUPPORT_ESCALATION"

    @patch('core.pipeline.ArabicInput')
    def test_run_pipeline_cleans_input_text(self, mock_arabic_input):
        mock_clean_text = MagicMock()
        mock_clean_text.text = "cleaned text"
        mock_arabic_input.return_value = mock_clean_text
        mock_loader = MockModelLoader()

        with patch('core.pipeline.predict_sentiment') as mock_sentiment, \
             patch('core.pipeline.predict_topic') as mock_topic, \
             patch('core.pipeline.predict_intent') as mock_intent:

            mock_sentiment.return_value = PredictionDetail(
                label=SentimentLabel.NEU,
                confidence=0.5,
                explanation="Test"
            )
            mock_topic.return_value = PredictionDetail(
                label=TopicLabel.CONTENT,
                confidence=0.5,
                explanation="Test"
            )
            mock_intent.return_value = PredictionDetail(
                label=ActionLabel.USER_REQUEST,
                confidence=0.5,
                explanation="Test"
            )

            run_pipeline("  dirty text with spaces  ", mock_loader)

            mock_arabic_input.assert_called_once_with(text="dirty text with spaces")

            mock_sentiment.assert_called_once_with("cleaned text", mock_loader)
            mock_topic.assert_called_once_with("cleaned text", mock_loader)
            mock_intent.assert_called_once_with("cleaned text", mock_loader)
