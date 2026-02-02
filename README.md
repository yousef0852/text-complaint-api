# Dialect Complaint Analysis API (MARBERT-v2)

An intelligent, multi-stage AI pipeline designed to analyze customer complaints in Saudi dialect and provide structured, explainable resolutions.

## 📊 System Architecture

The system follows a modular Clean Architecture pattern, separating HTTP concerns from the core AI orchestration logic.

- **API Layer**: FastAPI handles request validation and lifecycle management (Lifespan).
- **Orchestration Layer**: A central pipeline coordinates the sequence of preprocessing, inference, and mapping.
- **Inference Layer**: Leveraging MARBERT-v2 for high-accuracy Arabic NLP.
- **Logic Layer**: A deterministic rule engine maps model outputs to business-level actions.

## 🚀 Features & Capabilities

- **Multi-Model Inference**: Specialized models for Sentiment, Topic, and Intent classification.
- **Secure Authentication**: Integrated with Hugging Face Hub for private model access.
- **Deterministic Actions**: Logic-based routing for critical issues like POLICY_SECURITY or FINANCIAL_ESCALATION.
- **Confidence Guarding**: Automatic MANUAL_REVIEW override if any model falls below a 0.7 confidence threshold.
- **Clean Preprocessing**: Custom ArabicInput handler to strip whitespace and normalize inputs.

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| Framework | FastAPI |
| NLP Engine | Hugging Face Transformers |
| Model | MARBERT-v2 (Dialectal Arabic BERT) |
| Data Validation | Pydantic v2 |
| Server | Uvicorn with Auto-reload |

## 🔌 API Specification

### Predict Complaint

**POST** `/predict/`

#### Sample Request Body:
```json
{
  "text": "التطبيق يعلق وقت الدفع، حاولت أكثر من مرة وما يضبط معي"
}
```

#### Successful Response Structure:
```json
{
  "sentiment": {
    "label": "NEG",
    "confidence": 0.97,
    "explanation": "Sentiment analysis complete."
  },
  "topic": {
    "label": "TECHNICAL",
    "confidence": 1.0,
    "explanation": "Topic: LABEL_2 -> TECHNICAL"
  },
  "action": {
    "label": "TECH_SUPPORT_ESCALATION",
    "decision_source": "RULE_ENGINE"
  },
  "meta": {
    "model_version": "MARBERT-v2"
  }
}
```

## ⚙️ Installation & Setup

### Environment Setup:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Hugging Face Login (Required for private MARBERT models):
```bash
huggingface-cli login
```

### Run Server:
```bash
uvicorn main:app --reload
```

## 🏁 Status: Functional Foundation

The core inference engine and logic mapping are fully implemented and verified.
