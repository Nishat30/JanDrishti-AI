# JanDrishti AI — Sentiment & Emotion Module

**Team ZeninClan — Smart India Hackathon (SIH) 2026**

This repo is a working prototype of the **AI core** of JanDrishti AI, our SIH 2026 idea: a social media intelligence platform that ingests public posts across platforms (X, Telegram, Instagram, Facebook, Reddit, YouTube) and turns them into structured insight — emotion/sentiment, sarcasm-aware correction, demographic cohorts, trend detection, and cross-platform network analysis.

This module covers the **"Sentiment & Emotion"** piece of that pipeline: emotion classification fused with sarcasm detection, served as a single API.

## What it does

1. **Emotion classification** — tags input text with one or more of 28 emotions (GoEmotions taxonomy: joy, anger, confusion, gratitude, etc.), trained on Google's GoEmotions dataset (43K+ examples).
2. **Sarcasm detection** — a separate classifier trained on Sarcasm Corpus V2 estimates how likely the text is sarcastic.
3. **Fusion** — if sarcasm is detected on text that would otherwise read as positive (e.g. *"Oh great, another power cut, just what I needed"*), the emotion label is corrected to reflect the real sentiment (e.g. annoyance) instead of the literal, misleading one (admiration).
4. **API** — everything is wrapped in a FastAPI service (`api.py`) with a live browser demo and a `/analyze` endpoint, matching the Unified API layer in our full architecture.

## Status

Working CPU prototype: both models trained and verified (emotion micro-F1 0.50, sarcasm F1 0.75 on their respective test sets), fused pipeline tested end-to-end, API serving live. GPU fine-tuning scripts (`train_transformer.py`, `train_sarcasm_transformer.py`) are included for the production RoBERTa/DistilBERT versions but need a GPU (Colab/Kaggle) to run — this was built and verified in a CPU-only sandbox.

**Known limitation:** the sarcasm classifier's training data (Sarcasm Corpus V2) is long-form debate text, so very short inputs (under ~40 characters — typical of real tweets/chat messages) are out-of-distribution and can be misclassified. A length-based dampening guard is in place to reduce false positives; the real fix is training on short-text/social-media-style sarcasm data before this goes in front of judges with live short input. See `sarcasm_model.py` for details.

## Setup & run

```bash
pip install -r requirements.txt
python api.py
```

- Browser demo: http://localhost:8000
- Interactive API docs: http://localhost:8000/docs
- Direct call:
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Oh great, another power cut. Just what I needed."]}'
```

## Files

| File | Purpose |
|---|---|
| `api.py` | FastAPI service — the runnable demo |
| `inference.py` | Emotion model inference |
| `sarcasm_model.py` | Sarcasm model inference (baseline + real-checkpoint slot + short-text guard) |
| `sarcasm_fusion.py` | Combines the two into corrected emotion labels |
| `preprocess.py` / `sarcasm_preprocess.py` | Dataset loaders |
| `train_baseline.py` / `train_sarcasm_baseline.py` | CPU baseline training (TF-IDF + Logistic Regression) |
| `train_transformer.py` / `train_sarcasm_transformer.py` | GPU production training (RoBERTa/DistilBERT) |
| `data/` / `sarcasm_data/` | GoEmotions and Sarcasm Corpus V2 datasets |
| `*.joblib` | Trained baseline models, ready to load |

## Roadmap

Next planned module: BERTopic-based real-time trend detection, per the Technical Approach slide. Then link/network analysis (Neo4j) and the ingestion/streaming layer (Kafka).

---
Built for SIH 2026 as a proof-of-concept — not a final production system.