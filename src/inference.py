from pathlib import Path
import joblib

# Model path relative to project root
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "emotion_baseline.joblib"

bundle = joblib.load(MODEL_PATH)
vectorizer, clf, labels = bundle["vectorizer"], bundle["clf"], bundle["labels"]

def predict(texts, threshold=0.3, top_k=3):
    X = vectorizer.transform(texts)
    probs = clf.predict_proba(X)
    results = []
    for row in probs:
        ranked = sorted(zip(labels, row), key=lambda x: -x[1])
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
