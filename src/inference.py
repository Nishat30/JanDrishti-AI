import json
from pathlib import Path
import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "emotion_baseline.joblib"

# Optional per-label thresholds produced during transformer tuning
THRESHOLDS_PATH = ROOT_DIR / "models" / "emotion_transformer" / "thresholds.json"
if not THRESHOLDS_PATH.exists():
    THRESHOLDS_PATH = ROOT_DIR / "models" / "thresholds.json"

PER_LABEL_THRESHOLDS = None
if THRESHOLDS_PATH.exists():
    try:
        with open(THRESHOLDS_PATH, "r") as f:
            PER_LABEL_THRESHOLDS = json.load(f)
        print(f"[inference] Loaded per-label thresholds from {THRESHOLDS_PATH}")
    except Exception as e:
        print(f"[inference] Could not load thresholds from {THRESHOLDS_PATH}: {e}")

bundle = joblib.load(MODEL_PATH)
vectorizer, clf, labels = bundle["vectorizer"], bundle["clf"], bundle["labels"]


def predict(texts, threshold=0.3, top_k=3, use_per_label_thresholds=True):
    X = vectorizer.transform(texts)
    probs = clf.predict_proba(X)
    results = []
    for row in probs:
        ranked = sorted(zip(labels, row), key=lambda x: -x[1])
        if use_per_label_thresholds and PER_LABEL_THRESHOLDS:
            tags = [
                (lab, round(p, 3)) for lab, p in ranked
                if p >= PER_LABEL_THRESHOLDS.get(lab, threshold)
            ][:top_k]
        else:
            tags = [(lab, round(p, 3)) for lab, p in ranked if p >= threshold][:top_k]

        if not tags:
            tags = [(ranked[0][0], round(ranked[0][1], 3))]
        results.append(tags)
    return results


if __name__ == "__main__":
    samples = [
        "This is exactly what I needed today, thank you so much!",
        "Oh great, another power cut. Just what I needed.",  # sarcasm-flavored
        "I can't believe they cancelled the event without any warning.",
        "Not sure what's going on, can someone explain?",
    ]
    for text, tags in zip(samples, predict(samples)):
        print(f"{text!r}\n  -> {tags}\n")
