# Text Complaint API — Full Build Journey

A complete documentation of how this project was built from scratch, every decision made, every mistake encountered, and every lesson learned.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Phase 1: Foundation (January)](#phase-1-foundation-january)
3. [Phase 2: Containerization & Environment](#phase-2-containerization--environment)
4. [Phase 3: Observability & Reliability](#phase-3-observability--reliability)
5. [Phase 4: Deployment](#phase-4-deployment)
6. [Mistakes & Debugging Stories](#mistakes--debugging-stories)
7. [Architecture Deep Dive](#architecture-deep-dive)
8. [Every File Explained](#every-file-explained)
9. [Testing Scenarios](#testing-scenarios)
10. [What I'd Redesign](#what-id-redesign)

---

## Project Overview

### What is this?

An AI-powered REST API that takes an Arabic customer complaint as input and returns:
- **Sentiment**: Is the customer angry, neutral, or happy?
- **Topic**: What is the complaint about? (Financial, Technical, Security, Content)
- **Intent**: What does the customer want? (Report a bug, Make a request, General note)
- **Action**: What should we do about it? (Escalate, Create ticket, Archive, etc.)

### The Flow

```
Customer complaint (Arabic text)
        |
        v
  [Arabic Text Cleaning]     -- Remove diacritics, normalize letters
        |
        v
  [Sentiment Model]          -- MARBERT-v2 fine-tuned for Arabic
        |
        v
  [Topic Model]              -- MARBERT-v2 fine-tuned for complaint topics
        |
        v
  [Intent Model]             -- MARBERT-v2 fine-tuned for user intent
        |
        v
  [Rule Engine]              -- Deterministic rules based on all 3 outputs
        |
        v
  [Confidence Guard]         -- Override to MANUAL_REVIEW if confidence is low
        |
        v
  JSON Response with full analysis
```

### Why MARBERT-v2?

MARBERT is a BERT model pre-trained specifically on Arabic tweets and dialectal Arabic text. Standard BERT or multilingual models perform poorly on Saudi dialect because they were trained mostly on MSA (Modern Standard Arabic) or English. MARBERT understands colloquial expressions like "يعلق" (crashes/freezes) or "ما ردوا علي" (they didn't respond to me).

---

## Phase 1: Foundation (January)

### What we built

The core ML pipeline with clean architecture. The goal was: **structure first, then execution**.

### Project Structure Decision

We chose a layered architecture separating concerns:

```
interfaces/    -- HTTP layer (routes, schemas, middlewares)
core/          -- Business logic (pipeline, rule engine)
services/      -- ML model loading and inference
configs/       -- Settings, exceptions, logging
utils/         -- Text utilities
tests/         -- Unit tests
```

**Why this structure?** Because the pipeline (`core/pipeline.py`) should not know anything about HTTP. It takes text in and returns a result. The route (`predict_route.py`) is just an adapter that connects HTTP to the pipeline. This means we could swap FastAPI for Flask or even call the pipeline from a CLI without changing any core logic.

### Models Setup

Three separate models hosted on HuggingFace:
- `Ysfxjo/marbert-complaint-sentiment` — Sentiment (NEG, NEU, POS)
- `Ysfxjo/marbert-saudi-complaint-topic` — Topic (FINANCIAL, TECHNICAL, POLICY_SECURITY, CONTENT)
- `Ysfxjo/marbert-saudi-complaint-action` — Intent (GENERAL_NOTE, USER_REQUEST, REPORT_BUG)

Each model returns labels like `LABEL_0`, `LABEL_1`, etc. We created mapping dictionaries in each service to convert these to human-readable enums:

```python
mapping = {
    "LABEL_0": SentimentLabel.NEG,
    "LABEL_1": SentimentLabel.NEU,
    "LABEL_2": SentimentLabel.POS
}
```

### Rule Engine

Instead of using another ML model for action routing, we built a deterministic rule engine using Python's `match/case` (pattern matching). This was intentional:

- ML models can be unpredictable
- Business rules should be explicit and auditable
- You can explain exactly WHY an action was taken

The rules follow priority order:

| Priority | Condition | Action | Reasoning |
|----------|-----------|--------|-----------|
| 1 | Topic = POLICY_SECURITY | BLOCK_AND_REVIEW | Security issues always need human review |
| 2 | Topic = FINANCIAL + Sentiment = NEG | FINANCIAL_ESCALATION | Angry customer + money = urgent |
| 3 | Topic = TECH + Intent = REPORT_BUG | CREATE_JIRA_TICKET | Bug report goes to engineering |
| 4 | Topic = TECH + Sentiment = NEG | TECH_SUPPORT_ESCALATION | Frustrated tech user needs help |
| 5 | Topic = CONTENT + Intent = USER_REQUEST | CONTENT_MODIFICATION_QUEUE | User wants content changed |
| 6 | Sentiment = POS | AUTO_REPLY_THANK_YOU | Happy customer gets a thank you |
| 7 | Sentiment = NEU + Intent = GENERAL_NOTE | ARCHIVE_NOTE | Neutral note, just archive it |
| 8 | Default | GENERAL_SUPPORT_ROUTING | Everything else goes to general support |

### Arabic Text Cleaning

Arabic text has unique challenges:
- **Diacritics** (تشكيل): `مُحَمَّد` → `محمد`
- **Hamza variants**: `أ`, `إ`, `آ`, `ٱ` all normalize to `ا`
- **Taa marbuta**: `ة` → `ه`
- **Alef maqsura**: `ى` → `ي`
- **Extra whitespace**: collapsed to single space

This normalization happens BEFORE any model inference because the models were trained on normalized text.

### Pydantic Schemas

We used Pydantic v2 for:
- **Request validation**: `ComplaintRequest` requires `text` with min_length=1, max_length=5000
- **Response structure**: `ComplaintResponse` with typed fields
- **Enums**: `SentimentLabel`, `TopicLabel`, `ActionLabel` — ensures only valid values

### Unit Tests

Tests were written with `pytest` and `unittest.mock`. Key decision: **we mock the ML models** in tests because:
- Loading real models takes minutes and requires GPU/internet
- Tests should be fast and deterministic
- We test the logic, not the model accuracy

```python
class MockModelLoader:
    def __init__(self):
        self.sentiment_model = MagicMock(return_value=[[{"label":"LABEL_0","score":0.95}]])
        self.topic_model = MagicMock(return_value=[[{"label":"LABEL_2","score":0.90}]])
        self.action_model = MagicMock(return_value=[[{"label":"LABEL_1","score":0.85}]])
```

---

## Phase 2: Containerization & Environment

### Goal: "If I ship this, will it run?"

### Dockerfile

```dockerfile
FROM python:3.12.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home --shell /bin/bash app
RUN mkdir -p /app/logs && chown app:app /app/logs
USER app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Key decisions:
- **`python:3.12.11-slim`** — Slim image to reduce size (full image is ~1GB larger)
- **`PYTHONUNBUFFERED=1`** — Forces Python to print output immediately (critical for Docker logs)
- **`PYTHONDONTWRITEBYTECODE=1`** — Don't create `.pyc` files in container
- **`COPY requirements.txt` first** — Docker layer caching. If requirements don't change, this layer is cached and pip install is skipped on rebuild
- **Non-root user (`app`)** — Security best practice. Container runs as unprivileged user
- **`--no-cache-dir`** — Don't store pip cache in image (smaller image size)

### Docker Compose

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    restart: unless-stopped
```

- **`env_file: .env`** — Loads environment variables from `.env` file into the container
- **`restart: unless-stopped`** — If the container crashes, Docker restarts it automatically

### Environment Configuration

We created `configs/config.py` to centralize all settings:

```python
class Settings:
    HF_TOKEN: str = os.getenv("HF_TOKEN")
    SENTIMENT_THRESHOLD: float = float(os.getenv("SENTIMENT_THRESHOLD", "0.7"))
    TOPIC_THRESHOLD: float = float(os.getenv("TOPIC_THRESHOLD", "0.7"))
    INTENT_THRESHOLD: float = float(os.getenv("INTENT_THRESHOLD", "0.7"))
    ENABLE_CONFIDENCE_GUARDING: bool = os.getenv("ENABLE_CONFIDENCE_GUARDING", "true").lower() == "true"
    ENABLE_PREDICTION_LOGGING: bool = os.getenv("ENABLE_PREDICTION_LOGGING", "false").lower() == "true"
```

**Why environment variables?** Because the same code runs in different environments:
- **Local dev**: `.env` file with `HF_TOKEN=hf_xxx`, thresholds at 0.5 for testing
- **Docker**: `docker-compose.yml` loads `.env` automatically
- **Railway (production)**: Variables set in the dashboard, thresholds at 0.7 for stricter filtering

### .dockerignore

```
.venv/
.git/
__pycache__/
*.pyc
.env
logs/
.pytest_cache/
```

Without this file, `docker build` would copy the entire `.venv/` directory (hundreds of MBs) into the container, making the build slow and the image huge.

### .gitignore

Prevents sensitive/generated files from being committed:
- `.env` — Contains HF_TOKEN (secret)
- `logs/` — Runtime output, not source code
- `.venv/` — Virtual environment (installed per-machine)
- `*.json` — Prediction logs, data files

### Mistake: First Docker Build Failed

```
failed to resolve source metadata for docker.io/library/python:3.12.11-slim:
dial tcp: lookup registry-1.docker.io: no such host
```

**Cause**: DNS resolution failed. Docker couldn't reach the registry.
**Fix**: It was a network/WSL issue. Retrying after a few minutes worked.
**Lesson**: Network failures in Docker are common. Don't panic, check connectivity first.

---

## Phase 3: Observability & Reliability

### Goal: "How do I know my system is healthy?"

### Structured Logging (structlog)

We replaced `print()` statements with structured JSON logging.

**Before (print)**:
```
Loading model...
Sentiment Model loaded successfully!
```

**After (structlog)**:
```json
{"event": "sentiment_predicted", "label": "NEG", "confidence": 0.58, "request_id": "c72dc0a5-...", "path": "/predict", "method": "POST", "level": "info", "timestamp": "2026-02-27T22:23:57Z"}
```

**Why JSON?** Because:
- You can search logs by any field (`request_id`, `level`, `event`)
- Log aggregation tools (ELK, Datadog) parse JSON automatically
- Every log entry is self-contained with all context

### The Logging Pipeline

```python
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,    # 1. Add request context
        structlog.processors.add_log_level,          # 2. Add "level": "info"
        structlog.processors.TimeStamper(fmt="iso"), # 3. Add ISO timestamp
        structlog.processors.StackInfoRenderer(),    # 4. Render stack traces
        structlog.processors.format_exc_info,        # 5. Format exceptions
        structlog.processors.JSONRenderer(),          # 6. Output as JSON
    ],
)
```

Each log message passes through all 6 processors in order. The magic is `merge_contextvars` — it automatically adds `request_id`, `path`, and `method` to EVERY log without you writing them manually.

### Mistake: structlog `cache_logger_on_first_call`

```
TypeError: configure() got an unexpected keyword argument 'cache_logger_on_first_call'
```

**Cause**: We used `structlog>=23.0.0` in requirements, but pip installed a newer version (24.x) where this parameter was removed.
**Fix**: Removed the parameter. The functionality still works, it's just the default now.
**Lesson**: When using `>=` in requirements, newer versions may break APIs. Pin versions in production.

### Request ID Middleware

```python
class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        clear_contextvars()
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        bind_contextvars(request_id=request_id, path=request.url.path, method=request.method)

        start_time = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000

        logger.info("Request completed", status_code=response.status_code, duration_ms=round(duration_ms, 2))
        response.headers["x-request-id"] = request_id
        return response
```

**What it does**:
1. Every request gets a unique ID (or uses one from the header)
2. That ID is bound to ALL logs during that request's lifecycle
3. Request duration is measured and logged
4. The ID is returned in the response header

**Why this matters**: In production with hundreds of concurrent requests, you need to trace ONE request through ALL its logs. Without request_id, logs from different requests are mixed together and debugging is impossible.

### Custom Exception Handling

Three custom exceptions with structured error responses:

| Exception | When | Status Code | Error Code |
|-----------|------|-------------|------------|
| `ModelLoadError` | Model fails to download from HuggingFace | 503 | MODEL_LOAD_ERROR |
| `ConfigurationError` | Missing environment variable (e.g., HF_TOKEN) | 400 | CONFIG_ERROR |
| `PredictionError` | Model inference fails on input | 500 | PREDICTION_ERROR |

Each exception handler:
1. Logs the error with full context
2. Returns a structured JSON response (not a raw stack trace)
3. Uses appropriate HTTP status codes

**Why custom exceptions?** Because `Exception` tells you nothing. `ModelLoadError(model_name="sentiment", reason="401 Unauthorized")` tells you exactly what failed and why.

### Confidence Guarding

```python
if sentiment.confidence < threshold:
    action.label = "MANUAL_REVIEW"
    action.decision_source = "CONFIDENCE_THRESHOLD"
```

**The problem it solves**: ML models always return a prediction, even when they're not confident. A model that says "NEG with 35% confidence" is basically guessing. Without guarding, the rule engine would still route based on that guess.

**How it works**:
- Each model has a configurable confidence threshold (default 0.5)
- If ANY model's confidence is below its threshold, the action is overridden to `MANUAL_REVIEW`
- The `decision_source` changes from `RULE_ENGINE` to `CONFIDENCE_THRESHOLD`
- A warning log is emitted so you can track how often this happens

**Real example from our testing**:
```
Input: "حولت مبلغ ومارجع لي وخدمة العملاء ما ردوا علي"
Sentiment: NEU (confidence: 0.388) <-- Below 0.5 threshold
Topic: FINANCIAL (confidence: 1.0)
Intent: GENERAL_NOTE (confidence: 0.999)
Action: MANUAL_REVIEW (was going to be ARCHIVE_NOTE)
```

The model was unsure about the sentiment, so instead of archiving (which would be wrong — this is clearly a complaint), it sent it for human review.

### Prediction File Logging

When `ENABLE_PREDICTION_LOGGING=true`, every prediction is saved to `logs/predictions.json`:

```json
{
    "timestamp": "2026-02-27T22:21:25.715667",
    "input_text": "حولت مبلغ ومارجع لي",
    "response": {
        "sentiment": {"label": "NEG", "confidence": 0.58},
        "topic": {"label": "FINANCIAL", "confidence": 1.0},
        "action": {"label": "FINANCIAL_ESCALATION"}
    }
}
```

### Mistake: Arabic Text Showed as Question Marks

```json
"input_text": "??????? ???? ?? ?? ?????"
```

**Cause**: PowerShell on Windows doesn't send HTTP request bodies as UTF-8 by default. The Arabic characters were corrupted before reaching the server.
**Fix**: Use `[System.Text.Encoding]::UTF8.GetBytes()` to encode the body, or use the Swagger UI (`/docs`) which handles encoding correctly.
**Lesson**: The server code was correct (`ensure_ascii=False`, `encoding="utf-8"`). The problem was the client. Always verify encoding at both ends.

### Health Check

```python
@app.get("/health")
def health_check():
    return {"status": "ok"}
```

Simple but essential. Used by:
- Docker's restart policy to know if the container is alive
- Railway's deployment system to verify the service is running
- Monitoring systems to check availability

---

## Phase 4: Deployment

### Goal: "From my machine to somewhere else"

### Platform: Railway

We chose Railway because:
- Free tier ($5 credit)
- Connects directly to GitHub
- Auto-deploys on every push
- Supports Docker natively
- Provides a public URL

### Deployment Steps

1. Push code to GitHub
2. Connect GitHub repo to Railway
3. Add `HF_TOKEN` as an environment variable in Railway dashboard
4. Railway detects the Dockerfile and builds automatically
5. Generate a public domain in Settings > Networking

### What Happens on Deploy

```
1. Railway pulls the latest code from GitHub
2. Builds the Docker image (installs requirements ~15 min first time)
3. Starts the container
4. uvicorn starts → loads main.py → lifespan runs
5. ModelLoader downloads 3 models from HuggingFace (~2 min)
6. Server is ready to accept requests
```

### Mistake: WSL Crash During Docker Build

```
DockerDesktop/Wsl/ExecError: c:\windows\system32\wsl.exe --unmount
docker_data.vhdx: exit status 0xffffffff
```

**Cause**: WSL (Windows Subsystem for Linux) crashed during the Docker build. Docker Desktop on Windows depends on WSL2.
**Fix**: `wsl --shutdown` then restart Docker Desktop.
**Lesson**: Docker on Windows is less stable than on Linux/Mac. WSL crashes are common under heavy I/O (like installing large packages).

### Mistake: Docker Build `--no-cache` Was Slow

First build took **~15 minutes** because `--no-cache` forces reinstalling ALL pip packages (including torch which is ~2GB).

**Fix**: Use `docker-compose build` without `--no-cache`. Docker's layer caching means if `requirements.txt` hasn't changed, the pip install step is skipped entirely. Only the `COPY . .` step re-runs (takes seconds).

**Lesson**: Understand Docker layer caching. The order of commands in Dockerfile matters:
```dockerfile
COPY requirements.txt .          # Layer 1: changes rarely
RUN pip install -r requirements  # Layer 2: cached if Layer 1 unchanged
COPY . .                         # Layer 3: changes often (your code)
```

### Mistake: PowerShell `&&` Not Supported

```
docker-compose build && docker-compose up
The token '&&' is not a valid statement separator in this version.
```

**Cause**: PowerShell uses `;` not `&&` for command chaining.
**Fix**: Run commands separately, or use `docker-compose build; docker-compose up`.

---

## Architecture Deep Dive

### Request Lifecycle

```
1. HTTP Request arrives at uvicorn
2. RequestIdMiddleware:
   - Clears previous context
   - Generates/extracts request_id
   - Binds request_id + path + method to all future logs
   - Starts timer
3. FastAPI routing → matches /predict
4. Pydantic validates request body (ComplaintRequest)
   - If invalid → 422 response (middleware still logs it)
5. predict_route.py:
   - Gets model_loader from app.state
   - Calls run_pipeline(text, loader)
6. pipeline.py:
   - Cleans Arabic text
   - Runs 3 model predictions (sequential)
   - Applies rule engine
   - Checks confidence guard
   - Returns ComplaintResponse
7. predict_route.py:
   - Optionally saves prediction log
   - Returns JSON response
8. RequestIdMiddleware:
   - Calculates duration
   - Logs "Request completed" with status code and timing
   - Adds x-request-id to response headers
9. Response sent to client
```

### Error Lifecycle

```
1. Exception occurs (e.g., ModelLoadError)
2. FastAPI catches it → routes to registered exception handler
3. Exception handler:
   - Logs the error with full context
   - Returns structured JSON error response
4. RequestIdMiddleware:
   - Still runs (catches the response)
   - Logs "Request completed" with error status code
5. Client receives JSON error (not a raw stack trace)
```

### Middleware Order

```python
app.add_middleware(RequestIdMiddleware)   # Runs FIRST (outermost)
app.add_middleware(CORSMiddleware, ...)   # Runs SECOND
```

Middleware wraps like layers. The first added is the outermost:
```
Request → RequestIdMiddleware → CORSMiddleware → Route → CORSMiddleware → RequestIdMiddleware → Response
```

---

## Every File Explained

### `main.py` — Application Entry Point
- Creates FastAPI app with lifespan (startup/shutdown)
- On startup: checks HF_TOKEN, loads all 3 models
- Registers 3 exception handlers
- Adds RequestIdMiddleware and CORSMiddleware
- Includes the predict router
- Has `/health` and `/debug/env` endpoints

### `core/pipeline.py` — ML Pipeline + Rule Engine
- `predict_sentiment()`, `predict_topic()`, `predict_intent()` — Thin wrappers calling services
- `map_action()` — Pattern matching rule engine
- `run_pipeline()` — Orchestrates everything: clean → predict → rule → guard → respond
- Logs every step with structured data

### `services/model_loader.py` — Model Loading
- Detects CUDA/CPU availability
- Loads 3 HuggingFace pipelines with authentication
- Each model wrapped in try/except with `ModelLoadError`

### `services/sentiment_service.py`, `topic_service.py`, `action_service.py`
- Each takes text + model pipeline
- Runs inference, maps LABEL_X to enum
- Returns `PredictionDetail` with label, confidence, explanation, low_confidence flag

### `interfaces/api/predict_route.py` — HTTP Endpoint
- `POST /predict` route
- Gets model_loader via FastAPI dependency injection
- Calls pipeline, optionally logs prediction to file

### `interfaces/api/middlewares.py` — Request Middleware
- Generates unique request ID per request
- Binds context (request_id, path, method) to all logs
- Measures request duration
- Handles exceptions gracefully

### `interfaces/schemas/complaint.py` — Data Models
- `ComplaintRequest` — Input validation (text: 1-5000 chars)
- `PredictionDetail` — Label + confidence + explanation + low_confidence flag
- `ActionDetail` — Action label + decision source
- `ComplaintResponse` — Full response with sentiment, topic, intent, action, meta

### `interfaces/schemas/enums.py` — Label Enums
- `SentimentLabel`: NEG, NEU, POS
- `TopicLabel`: CONTENT, TECHNICAL, POLICY_SECURITY, FINANCIAL
- `ActionLabel`: REPORT_BUG, USER_REQUEST, GENERAL_NOTE

### `configs/config.py` — Settings
- All settings from environment variables with defaults
- Thresholds, feature flags, tokens

### `configs/exceptions.py` — Custom Exceptions
- `ComplaintAPIException` — Base with message, error_code, details
- `ModelLoadError` — Model name + reason
- `PredictionError` — Input text + reason
- `ConfigurationError` — Config key + reason

### `configs/logging.py` — Logging Setup
- Configures structlog with JSON output
- 6 processors: contextvars → log level → timestamp → stack → exceptions → JSON
- `get_logger(name)` factory function

### `utils/text_utils.py` — Arabic Text Normalization
- Pydantic model with field validator
- Strips diacritics, normalizes hamza/taa/alef, collapses whitespace

### `tests/test_api.py` — API Tests
- Tests `/predict` endpoint with mocked pipeline
- Tests empty text validation (422)

### `tests/test_pipeline.py` — Pipeline Tests
- Tests each prediction function returns correct type
- Tests all 8 rule engine cases
- Tests full pipeline orchestration
- Tests input text cleaning

---

## Testing Scenarios

### Scenario 1: Normal Complaint
```
Input:  "حولت مبلغ ومارجع لي وخدمة العملاء ما ردوا علي"
Result: Sentiment=NEG, Topic=FINANCIAL, Intent=USER_REQUEST
Action: FINANCIAL_ESCALATION (Rule Engine)
```
The rule engine matched: Financial topic + Negative sentiment = Escalate.

### Scenario 2: Low Confidence → Manual Review
```
Input:  "التطبيق يعلق كل ما افتحه ومحد رد علي" (sent with wrong encoding)
Result: Sentiment=NEU (confidence=0.388), Topic=FINANCIAL, Intent=GENERAL_NOTE
Action: MANUAL_REVIEW (Confidence Threshold)
```
Sentiment confidence was 38.8% (below 50% threshold). The system overrode the action to MANUAL_REVIEW instead of making a potentially wrong automated decision.

### Scenario 3: Empty Text
```
Input:  {"text": ""}
Result: HTTP 422
Body:   {"detail": [{"type": "string_too_short", "msg": "String should have at least 1 character"}]}
```
Pydantic validation caught the empty string before it reached the pipeline.

### Scenario 4: Missing Request Body
```
Input:  POST /predict with no body
Result: HTTP 422
Body:   {"detail": [{"msg": "Field required", "type": "missing"}]}
```

### Scenario 5: Health Check
```
Input:  GET /health
Result: HTTP 200
Body:   {"status": "ok"}
Headers: x-request-id: 04970750-4ae5-4ae2-b09f-dc9b1e761ad4
```
Middleware still runs on health checks — every request gets a request_id.

---

## What I'd Redesign

### 1. Prediction Logging
Currently reads and rewrites the entire JSON file on every request. At scale, this would be a bottleneck. Better approach: append to a log file, or use a database.

### 2. `/debug/env` Endpoint
Exposes token information. Should be removed or protected with authentication before production.

### 3. CORS Configuration
`allow_origins=["*"]` allows any website to call the API. Should be restricted to specific domains.

### 4. Model Loading on Startup
Models are downloaded from HuggingFace on every container start. Should cache models in a Docker volume or bake them into the image.

### 5. No Authentication
Anyone can call the API. Should add API key authentication at minimum.

### 6. Sequential Model Inference
The 3 models run one after another. They could run in parallel (asyncio) since they're independent, cutting latency by ~60%.

---

## Tech Stack Summary

| Component | Technology | Why |
|-----------|------------|-----|
| Framework | FastAPI | Async, auto-docs, Pydantic integration |
| NLP Models | MARBERT-v2 via HuggingFace | Best Arabic dialect model available |
| Validation | Pydantic v2 | Type safety, auto-validation |
| Logging | structlog | Structured JSON, context binding |
| Container | Docker + docker-compose | Reproducible environments |
| Server | Uvicorn | ASGI, high performance |
| Deployment | Railway | Simple, free tier, GitHub integration |
| Testing | pytest + unittest.mock | Fast, deterministic tests |
