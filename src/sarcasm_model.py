import os
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRANSFORMER_PATH = str(ROOT_DIR / "models" / "sarcasm_transformer")
MODEL_PATH = os.environ.get("SARCASM_MODEL_PATH", DEFAULT_TRANSFORMER_PATH)
BASELINE_MODEL_PATH = ROOT_DIR / "models" / "sarcasm_baseline.joblib"

MIN_RELIABLE_CHARS = 40  # shortest example in Sarcasm Corpus V2's training data

_backend = None  # lazily resolved: "transformer" | "baseline"
_transformer_model = None
_transformer_tokenizer = None
_baseline = None


def _try_load_transformer():
    global _transformer_model, _transformer_tokenizer
    if not Path(MODEL_PATH).exists():
        return False
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _transformer_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _transformer_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _transformer_model.eval()
        return True
    except Exception as e:
        print(f"[sarcasm_model] Could not load transformer at {MODEL_PATH}: {e}")
        return False


def _load_baseline():
    global _baseline
    import joblib
    _baseline = joblib.load(BASELINE_MODEL_PATH)


def _ensure_backend():
    global _backend
    if _backend is not None:
        return
    if _try_load_transformer():
        _backend = "transformer"
        print(f"[sarcasm_model] Using fine-tuned transformer at {MODEL_PATH}")
    else:
        _load_baseline()
        _backend = "baseline"
        print("[sarcasm_model] No transformer checkpoint found — "
              f"using TF-IDF+LogReg baseline ({BASELINE_MODEL_PATH.name}). "
              "Run train_sarcasm_transformer.py on a GPU, or set "
              "SARCASM_MODEL_PATH to your existing Sarcasm-V2 checkpoint, "
              "for the real model.")


def predict_sarcasm_proba(texts: List[str]) -> List[float]:
    """Returns P(sarcastic) for each text — drop-in replacement for
    sarcasm_fusion.py's dummy_sarcasm_predict.

    Guard: Sarcasm Corpus V2 (this baseline's training data) is long-form
    debate-forum text — avg 279 chars, shortest example 38 chars. Very short
    inputs (typical of tweets/chat messages) are out-of-distribution and the
    model's intercept defaults toward "sarcastic" with no real signal (even
    an empty string scores ~0.65). We dampen toward neutral (0.5) for short
    inputs rather than trusting an unfounded confident score.
    """
    _ensure_backend()
    raw = _raw_predict(texts)
    out = []
    for text, prob in zip(texts, raw):
        n_chars = len(text.strip())
        if n_chars < MIN_RELIABLE_CHARS:
            # Linearly fade the score toward 0.5 (no opinion) as length drops
            # below the reliable range, instead of a hard cutoff.
            confidence = n_chars / MIN_RELIABLE_CHARS
            prob = 0.5 + (prob - 0.5) * confidence
        out.append(prob)
    return out


def _raw_predict(texts: List[str]) -> List[float]:
    if _backend == "transformer":
        import torch
        enc = _transformer_tokenizer(texts, truncation=True, padding=True,
                                      max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = _transformer_model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]  # class 1 = sarc
        return probs.tolist()
    else:
        X = _baseline["vectorizer"].transform(texts)
        return _baseline["clf"].predict_proba(X)[:, 1].tolist()


if __name__ == "__main__":
    samples = [
        "This is exactly what I needed today, thank you so much!",
        "Oh great, another power cut. Just what I needed.",
        "Sure, love that the train is delayed again.",
    ]
    for text, prob in zip(samples, predict_sarcasm_proba(samples)):
        print(f"{prob:.3f}  {text!r}")
