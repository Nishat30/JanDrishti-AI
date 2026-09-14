# JanDrishti AI — Sentiment & Emotion Module

**Team ZeninClan — Smart India Hackathon (SIH) 2026**

This repo is a working prototype of the **AI core** of JanDrishti AI, our SIH 2026 idea: a social media intelligence platform that ingests public posts across platforms (X, Telegram, Instagram, Facebook, Reddit, YouTube) and turns them into structured insight — emotion/sentiment, sarcasm-aware correction, demographic cohorts, trend detection, and cross-platform network analysis.

This module covers the **"Sentiment & Emotion"** piece of that pipeline: emotion classification fused with sarcasm detection, served as a single API.

## What it does

1. **Emotion classification** — tags input text with one or more of 28 emotions (GoEmotions taxonomy: joy, anger, confusion, gratitude, etc.), trained on Google's GoEmotions dataset (43K+ examples).
2. **Sarcasm detection** — a separate classifier trained on Sarcasm Corpus V2 estimates how likely the text is sarcastic.
3. **Fusion** — if sarcasm is detected on text that would otherwise read as positive (e.g. *"Oh great, another power cut, just what I needed"*), the emotion label is corrected to reflect the real sentiment (e.g. annoyance) instead of the literal, misleading one (admiration).
4. **API** — everything is wrapped in a FastAPI service (`api.py`) with a live browser demo and a `/analyze` endpoint, matching the Unified API layer in our full architecture.
<img width="2816" height="1204" alt="Gemini_Generated_Image_fplf1kfplf1kfplf" src="https://github.com/user-attachments/assets/16f95ef7-717d-4e53-b0e8-dab20e92439a" />


## Status

Working CPU prototype: both models trained and verified (emotion micro-F1 0.50, sarcasm F1 0.75 on their respective test sets), fused pipeline tested end-to-end, API serving live. GPU fine-tuning scripts (`train_transformer.py`, `train_sarcasm_transformer.py`) are included for the production RoBERTa/DistilBERT versions but need a GPU (Colab/Kaggle) to run — this was built and verified in a CPU-only sandbox.

**Known limitation:** the sarcasm classifier's training data (Sarcasm Corpus V2) is long-form debate text, so very short inputs (under ~40 characters — typical of real tweets/chat messages) are out-of-distribution and can be misclassified. A length-based dampening guard is in place to reduce false positives; the real fix is training on short-text/social-media-style sarcasm data before this goes in front of judges with live short input. See `sarcasm_model.py` for details.

## Setup & run

```bash
pip install -r requirements.txt
python main.py
```

- Browser demo: http://localhost:8000
- Interactive API docs: http://localhost:8000/docs
- Direct call:
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Oh great, another power cut. Just what I needed."]}'
```

## Structure & Files

```
JanDrishti-AI/
├── data/                             # GoEmotions dataset (train, dev, test, emotions.txt)
├── sarcasm_data/                     # Sarcasm Corpus V2 dataset
├── models/                           # Stored baseline models & checkpoint outputs
│   ├── emotion_baseline.joblib
│   └── sarcasm_baseline.joblib
├── src/                              # Source code directory
│   ├── api.py                        # FastAPI service & embedded HTML browser demo
│   ├── inference.py                  # Emotion model inference module
│   ├── preprocess.py                 # Emotion dataset preprocessing & loader
│   ├── sarcasm_fusion.py             # Emotion + Sarcasm fusion rules engine
│   ├── sarcasm_model.py              # Sarcasm model loader & length-dampening guard
│   ├── sarcasm_preprocess.py         # Sarcasm dataset preprocessing & loader
│   ├── train_baseline.py             # CPU baseline training (TF-IDF + LogReg for emotions)
│   ├── train_sarcasm_baseline.py     # CPU baseline training (TF-IDF + LogReg for sarcasm)
│   ├── train_sarcasm_transformer.py # Fine-tuning script for RoBERTa/DistilBERT sarcasm
│   └── train_transformer.py          # Fine-tuning script for RoBERTa/DistilBERT emotion
├── main.py                           # Application entrypoint to run FastAPI service
├── requirements.txt                  # Python dependencies
├── README.md                         # Project overview and instructions
└── .gitignore                        # Git ignore rules
```

## Roadmap

Next planned module: BERTopic-based real-time trend detection, per the Technical Approach slide. Then link/network analysis (Neo4j) and the ingestion/streaming layer (Kafka).

---
Built for SIH 2026 as a proof-of-concept — not a final production system.
