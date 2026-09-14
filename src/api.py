"""
JanDrishti AI — Sentiment & Emotion Module, unified service.

This is the "whole working model": one process that loads the emotion
classifier and sarcasm classifier once, and exposes a single endpoint that
runs the full pipeline (emotion tagging -> sarcasm gate -> fused label) on
any input text. Matches your slide's "Unified API (FastAPI)" layer sitting
in front of the AI Analytics modules.

Run:
    python main.py
    # or:
    python -m src.api
    # then in another terminal:
    curl -X POST http://localhost:8000/analyze \
      -H "Content-Type: application/json" \
      -d '{"texts": ["Oh great, another power cut. Just what I needed."]}'

Or open http://localhost:8000/docs for interactive Swagger UI, or
http://localhost:8000/ for a minimal browser demo.
"""
from typing import List, Tuple
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    from .inference import predict as emotion_predict
    from .sarcasm_model import predict_sarcasm_proba
    from .sarcasm_fusion import fuse
except ImportError:
    from src.inference import predict as emotion_predict
    from src.sarcasm_model import predict_sarcasm_proba
    from src.sarcasm_fusion import fuse

app = FastAPI(
    title="JanDrishti AI — Sentiment & Emotion Module",
    description="Emotion classification (GoEmotions, 28 classes) fused with "
                "sarcasm detection (Sarcasm Corpus V2) into one endpoint.",
    version="0.1.0",
)


class AnalyzeRequest(BaseModel):
    texts: List[str]
    sarcasm_threshold: float = 0.6


class EmotionScore(BaseModel):
    label: str
    score: float


class AnalyzeResult(BaseModel):
    text: str
    raw_emotions: List[EmotionScore]
    sarcasm_probability: float
    sarcasm_detected: bool
    final_emotions: List[EmotionScore]


@app.post("/analyze", response_model=List[AnalyzeResult])
def analyze(req: AnalyzeRequest):
    fused = fuse(req.texts, emotion_predict, predict_sarcasm_proba,
                 sarcasm_threshold=req.sarcasm_threshold)
    out = []
    for r in fused:
        out.append(AnalyzeResult(
            text=r.text,
            raw_emotions=[EmotionScore(label=l, score=float(s)) for l, s in r.raw_emotions],
            sarcasm_probability=float(r.sarcasm_prob),
            sarcasm_detected=r.sarcasm_detected,
            final_emotions=[EmotionScore(label=l, score=float(s)) for l, s in r.final_emotions],
        ))
    return out


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def demo_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <title>JanDrishti AI — Sentiment &amp; Emotion Demo</title>
      <style>
        body { font-family: system-ui, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; }
        textarea { width: 100%; height: 80px; font-size: 15px; padding: 8px; box-sizing: border-box; }
        button { margin-top: 10px; padding: 8px 16px; font-size: 15px; cursor: pointer; }
        .result { margin-top: 20px; padding: 12px; border: 1px solid #ddd; border-radius: 8px; }
        .sarcastic { border-color: #e07b39; background: #fff7f0; }
        .tag { display: inline-block; background: #eee; border-radius: 12px; padding: 2px 10px; margin: 2px; font-size: 13px; }
      </style>
    </head>
    <body>
      <h2>JanDrishti AI — Sentiment &amp; Emotion Module</h2>
      <p>Enter one line of text per row.</p>
      <textarea id="input">Oh great, another power cut. Just what I needed.
This is exactly what I needed today, thank you so much!</textarea>
      <br><button onclick="analyze()">Analyze</button>
      <div id="results"></div>
      <script>
        async function analyze() {
          const texts = document.getElementById('input').value.split('\\n').filter(t => t.trim());
          const res = await fetch('/analyze', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({texts})
          });
          const data = await res.json();
          const container = document.getElementById('results');
          container.innerHTML = data.map(r => `
            <div class="result ${r.sarcasm_detected ? 'sarcastic' : ''}">
              <strong>${r.text}</strong><br>
              Sarcasm probability: ${r.sarcasm_probability.toFixed(2)}
              ${r.sarcasm_detected ? ' — flagged as sarcastic' : ''}<br>
              Final emotions: ${r.final_emotions.map(e => `<span class="tag">${e.label} ${e.score.toFixed(2)}</span>`).join(' ')}
            </div>
          `).join('');
        }
      </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
