# Text Complaint Mini Pipeline API

A small, clean AI pipeline API that analyzes customer complaint text and produces
structured, explainable outputs.

## Overview

This project receives free-form complaint text (Arabic/English mixed) and returns:
- Sentiment classification
- Topic classification
- High-level intent
- A deterministic, explainable final action

The system is designed for clarity, reliability, and explainability rather than
production-scale performance.

## System Scope

- Single FastAPI service
- In-process model inference
- Local file system for optional logging
- No database
- No external services

This is a foundation project focused on system design and correctness.

## High-Level Pipeline

1. Input validation
2. Text preprocessing
3. Sentiment classification
4. Topic classification
5. Intent classification
6. Rule-based action mapping
7. Structured JSON response

## API Interface (Conceptual)

- Endpoint: `POST /predict`
- Input: `{ "text": "complaint text" }`
- Output: structured JSON with predictions, confidence scores, and explanations

## Status

🚧 Early development — structure and foundations first.
Business logic and models will be added incrementally.
