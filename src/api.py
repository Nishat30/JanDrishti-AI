"""
JanDrishti AI — Sentiment & Emotion Module, unified service & dashboard UI.

Runs the full pipeline (emotion tagging -> sarcasm gate -> fused label) on
any input text and presents a state-of-the-art interactive analytics dashboard.

Run:
    python main.py
    # or:
    python -m src.api
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
                "sarcasm detection (Sarcasm Corpus V2) into one unified API.",
    version="0.2.0",
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
    return {"status": "ok", "service": "JanDrishti AI Core", "version": "0.2.0"}


@app.get("/", response_class=HTMLResponse)
def demo_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>JanDrishti AI — Social Media Intelligence Core</title>
      <link rel="preconnect" href="https://fonts.googleapis.com">
      <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
      <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
      <style>
        :root {
          --bg-dark: #0b0f19;
          --panel-bg: rgba(18, 26, 43, 0.75);
          --panel-border: rgba(255, 255, 255, 0.08);
          --accent-purple: #8b5cf6;
          --accent-indigo: #6366f1;
          --accent-pink: #ec4899;
          --accent-sarcasm: #f97316;
          --text-primary: #f3f4f6;
          --text-secondary: #9ca3af;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
          background-color: var(--bg-dark);
          background-image: 
            radial-gradient(at 10% 10%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
            radial-gradient(at 90% 90%, rgba(236, 72, 153, 0.12) 0px, transparent 50%);
          color: var(--text-primary);
          min-height: 100vh;
          padding: 2rem 1rem;
        }

        .container {
          max-width: 1100px;
          margin: 0 auto;
        }

        header {
          text-align: center;
          margin-bottom: 2.5rem;
        }

        .badge-sih {
          display: inline-block;
          padding: 0.35rem 0.9rem;
          background: rgba(139, 92, 246, 0.15);
          border: 1px solid rgba(139, 92, 246, 0.3);
          border-radius: 9999px;
          color: #c4b5fd;
          font-size: 0.825rem;
          font-weight: 600;
          letter-spacing: 0.05em;
          text-transform: uppercase;
          margin-bottom: 0.75rem;
        }

        h1 {
          font-size: 2.5rem;
          font-weight: 800;
          background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          margin-bottom: 0.5rem;
        }

        p.subtitle {
          color: var(--text-secondary);
          font-size: 1.05rem;
        }

        .grid {
          display: grid;
          grid-template-columns: 1fr;
          gap: 1.5rem;
        }

        @media (min-width: 900px) {
          .grid-two {
            grid-template-columns: 1fr 1fr;
          }
        }

        .card {
          background: var(--panel-bg);
          backdrop-filter: blur(16px);
          -webkit-backdrop-filter: blur(16px);
          border: 1px solid var(--panel-border);
          border-radius: 1rem;
          padding: 1.5rem;
          box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
        }

        .card-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 1rem;
        }

        .card-title {
          font-size: 1.1rem;
          font-weight: 600;
          color: #f8fafc;
        }

        textarea {
          width: 100%;
          height: 110px;
          background: rgba(11, 15, 25, 0.7);
          border: 1px solid var(--panel-border);
          border-radius: 0.75rem;
          padding: 0.85rem;
          color: #f8fafc;
          font-family: inherit;
          font-size: 0.95rem;
          resize: vertical;
          outline: none;
          transition: border-color 0.2s;
        }

        textarea:focus {
          border-color: var(--accent-indigo);
        }

        .presets {
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
          margin-top: 0.75rem;
          margin-bottom: 1rem;
        }

        .preset-btn {
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          color: #d1d5db;
          padding: 0.35rem 0.75rem;
          border-radius: 0.5rem;
          font-size: 0.8rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .preset-btn:hover {
          background: rgba(139, 92, 246, 0.2);
          border-color: rgba(139, 92, 246, 0.4);
          color: #fff;
        }

        .btn-submit {
          width: 100%;
          background: linear-gradient(135deg, var(--accent-indigo), var(--accent-purple));
          color: #ffffff;
          font-weight: 600;
          font-size: 1rem;
          padding: 0.85rem;
          border: none;
          border-radius: 0.75rem;
          cursor: pointer;
          transition: transform 0.1s, box-shadow 0.2s;
          box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
        }

        .btn-submit:hover {
          transform: translateY(-1px);
          box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45);
        }

        .stats-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 1rem;
          margin-top: 1.5rem;
          margin-bottom: 1.5rem;
        }

        .stat-box {
          background: rgba(15, 23, 42, 0.6);
          border: 1px solid var(--panel-border);
          border-radius: 0.75rem;
          padding: 1rem;
          text-align: center;
        }

        .stat-value {
          font-size: 1.6rem;
          font-weight: 700;
          color: #a7f3d0;
        }

        .stat-label {
          font-size: 0.775rem;
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin-top: 0.25rem;
        }

        .result-item {
          background: rgba(15, 23, 42, 0.6);
          border: 1px solid var(--panel-border);
          border-radius: 0.75rem;
          padding: 1.25rem;
          margin-bottom: 1rem;
          transition: border-color 0.2s;
        }

        .result-item.sarcastic {
          border-color: rgba(249, 115, 22, 0.5);
          background: linear-gradient(135deg, rgba(249, 115, 22, 0.08) 0%, rgba(15, 23, 42, 0.7) 100%);
        }

        .result-text {
          font-size: 1.05rem;
          font-weight: 600;
          color: #f1f5f9;
          margin-bottom: 0.75rem;
        }

        .badge {
          display: inline-flex;
          align-items: center;
          padding: 0.25rem 0.65rem;
          border-radius: 9999px;
          font-size: 0.8rem;
          font-weight: 500;
          margin-right: 0.4rem;
          margin-bottom: 0.4rem;
        }

        .badge-emotion {
          background: rgba(99, 102, 241, 0.2);
          border: 1px solid rgba(99, 102, 241, 0.4);
          color: #c7d2fe;
        }

        .badge-sarcasm {
          background: rgba(249, 115, 22, 0.2);
          border: 1px solid rgba(249, 115, 22, 0.5);
          color: #fdba74;
        }

        .badge-flipped {
          background: rgba(236, 72, 153, 0.25);
          border: 1px solid rgba(236, 72, 153, 0.5);
          color: #fbcfe8;
          font-weight: 600;
        }

        .chart-container {
          position: relative;
          height: 240px;
          width: 100%;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <header>
          <div class="badge-sih">Smart India Hackathon 2026 — Team ZeninClan</div>
          <h1>JanDrishti AI Intelligence Engine</h1>
          <p class="subtitle">Real-time Sarcasm-Aware Emotion &amp; Sentiment Analytics Module</p>
        </header>

        <div class="card" style="margin-bottom: 1.5rem;">
          <div class="card-header">
            <span class="card-title">Analyze Social Media Text</span>
            <span style="font-size: 0.8rem; color: var(--text-secondary);">Supports batch multi-line input</span>
          </div>
          <textarea id="inputText" placeholder="Type or paste social media posts here (one per line)..."></textarea>
          
          <div class="presets">
            <span style="font-size: 0.8rem; color: var(--text-secondary); align-self: center;">Try sample:</span>
            <button class="preset-btn" onclick="setPreset(1)">⚡ Power Outage Sarcasm</button>
            <button class="preset-btn" onclick="setPreset(2)">❤️ Appreciation</button>
            <button class="preset-btn" onclick="setPreset(3)">😡 Event Delay</button>
            <button class="preset-btn" onclick="setPreset(4)">❓ Confusion</button>
          </div>

          <button class="btn-submit" onclick="runAnalysis()">Run Intelligence Pipeline</button>
        </div>

        <div id="analyticsSection" style="display: none;">
          <div class="stats-grid">
            <div class="stat-box">
              <div class="stat-value" id="statTotal">0</div>
              <div class="stat-label">Texts Analyzed</div>
            </div>
            <div class="stat-box">
              <div class="stat-value" id="statSarcasm" style="color: #fdba74;">0%</div>
              <div class="stat-label">Sarcasm Detected</div>
            </div>
            <div class="stat-box">
              <div class="stat-value" id="statTopEmotion" style="color: #c7d2fe;">-</div>
              <div class="stat-label">Dominant Emotion</div>
            </div>
          </div>

          <div class="grid grid-two" style="margin-bottom: 1.5rem;">
            <div class="card">
              <div class="card-header">
                <span class="card-title">Emotion Distribution</span>
              </div>
              <div class="chart-container">
                <canvas id="emotionChart"></canvas>
              </div>
            </div>

            <div class="card">
              <div class="card-header">
                <span class="card-title">Sarcasm Detection Meter</span>
              </div>
              <div class="chart-container">
                <canvas id="sarcasmChart"></canvas>
              </div>
            </div>
          </div>

          <div class="card">
            <div class="card-header">
              <span class="card-title">Detailed Pipeline Results</span>
            </div>
            <div id="resultsList"></div>
          </div>
        </div>
      </div>

      <script>
        let emotionChartInst = null;
        let sarcasmChartInst = null;

        const samples = {
          1: "Oh great, another power cut. Just what I needed today!\\nThis is exactly what I needed today, thank you so much!",
          2: "Really grateful for the quick response from the municipality team. Thank you!",
          3: "I can't believe they cancelled the public transport event without any warning.",
          4: "Not sure what's going on with the new policy, can someone explain?"
        };

        function setPreset(id) {
          document.getElementById('inputText').value = samples[id];
          runAnalysis();
        }

        async function runAnalysis() {
          const raw = document.getElementById('inputText').value.trim();
          if (!raw) return;
          const texts = raw.split('\\n').filter(t => t.trim().length > 0);
          if (texts.length === 0) return;

          const res = await fetch('/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({texts})
          });
          const data = await res.json();
          renderDashboard(data);
        }

        function renderDashboard(results) {
          document.getElementById('analyticsSection').style.display = 'block';

          // 1. Stats
          const total = results.length;
          const sarcCount = results.filter(r => r.sarcasm_detected).length;
          const sarcPct = Math.round((sarcCount / total) * 100);

          document.getElementById('statTotal').innerText = total;
          document.getElementById('statSarcasm').innerText = `${sarcPct}%`;

          // Collect emotion counts
          const emoCounts = {};
          results.forEach(r => {
            r.final_emotions.forEach(e => {
              emoCounts[e.label] = (emoCounts[e.label] || 0) + 1;
            });
          });

          let topEmotion = '-';
          let maxC = 0;
          for (const [k, v] of Object.entries(emoCounts)) {
            if (v > maxC) { maxC = v; topEmotion = k; }
          }
          document.getElementById('statTopEmotion').innerText = topEmotion;

          // 2. Charts
          renderEmotionChart(emoCounts);
          renderSarcasmChart(sarcCount, total - sarcCount);

          // 3. Results list
          const container = document.getElementById('resultsList');
          container.innerHTML = results.map(r => `
            <div class="result-item ${r.sarcasm_detected ? 'sarcastic' : ''}">
              <div class="result-text">${escapeHtml(r.text)}</div>
              <div>
                <span class="badge badge-sarcasm">
                  Sarcasm: ${(r.sarcasm_probability * 100).toFixed(0)}% 
                  ${r.sarcasm_detected ? ' (Flagged Sarcastic)' : ' (Literal)'}
                </span>
              </div>
              <div style="margin-top: 0.5rem; font-size: 0.85rem; color: var(--text-secondary);">
                Raw Emotions: ${r.raw_emotions.map(e => `<span class="badge badge-emotion">${e.label} ${(e.score*100).toFixed(0)}%</span>`).join('')}
              </div>
              <div style="margin-top: 0.4rem; font-size: 0.85rem; color: var(--text-secondary);">
                Final Fused Result: ${r.final_emotions.map(e => `
                  <span class="badge ${r.sarcasm_detected ? 'badge-flipped' : 'badge-emotion'}">
                    ${e.label} ${(e.score*100).toFixed(0)}%
                  </span>`).join('')}
              </div>
            </div>
          `).join('');
        }

        function renderEmotionChart(emoCounts) {
          const labels = Object.keys(emoCounts);
          const values = Object.values(emoCounts);
          const ctx = document.getElementById('emotionChart').getContext('2d');

          if (emotionChartInst) emotionChartInst.destroy();

          emotionChartInst = new Chart(ctx, {
            type: 'bar',
            data: {
              labels: labels,
              datasets: [{
                label: 'Emotion Frequency',
                data: values,
                backgroundColor: 'rgba(99, 102, 241, 0.6)',
                borderColor: 'rgba(129, 140, 248, 1)',
                borderWidth: 1.5,
                borderRadius: 6
              }]
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              plugins: { legend: { display: false } },
              scales: {
                y: { beginAtZero: true, ticks: { precision: 0, color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
              }
            }
          });
        }

        function renderSarcasmChart(sarc, nonSarc) {
          const ctx = document.getElementById('sarcasmChart').getContext('2d');

          if (sarcasmChartInst) sarcasmChartInst.destroy();

          sarcasmChartInst = new Chart(ctx, {
            type: 'doughnut',
            data: {
              labels: ['Sarcastic', 'Literal / Non-Sarcastic'],
              datasets: [{
                data: [sarc, nonSarc],
                backgroundColor: ['rgba(249, 115, 22, 0.7)', 'rgba(99, 102, 241, 0.5)'],
                borderColor: ['rgba(249, 115, 22, 1)', 'rgba(99, 102, 241, 1)'],
                borderWidth: 1.5
              }]
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: { position: 'bottom', labels: { color: '#94a3b8' } }
              }
            }
          });
        }

        function escapeHtml(str) {
          return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        }

        // Auto-run first sample on load
        window.addEventListener('DOMContentLoaded', () => {
          setPreset(1);
        });
      </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
