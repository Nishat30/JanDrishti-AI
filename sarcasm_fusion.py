from dataclasses import dataclass
from typing import Callable, List, Tuple

POSITIVE_VALENCE = {
    "admiration", "amusement", "approval", "caring", "desire", "excitement",
    "gratitude", "joy", "love", "optimism", "pride", "relief",
}
# Simple reversal target when sarcasm flips a positive-surface reading
FLIP_TO = "annoyance"

@dataclass
class FusedResult:
    text: str
    raw_emotions: List[Tuple[str, float]]
    sarcasm_prob: float
    sarcasm_detected: bool
    final_emotions: List[Tuple[str, float]]

def fuse(
    texts: List[str],
    emotion_predict_fn: Callable[[List[str]], List[List[Tuple[str, float]]]],
    sarcasm_predict_fn: Callable[[List[str]], List[float]],
    sarcasm_threshold: float = 0.6,
) -> List[FusedResult]:
    """
    emotion_predict_fn: texts -> list of [(label, prob), ...] (e.g. inference.predict)
    sarcasm_predict_fn: texts -> list of sarcasm probabilities in [0, 1]
                         (wire this to your Sarcasm-V2 RoBERTa model's inference fn)
    """
    raw_emotions = emotion_predict_fn(texts)
    sarcasm_probs = sarcasm_predict_fn(texts)

    results = []
    for text, emos, s_prob in zip(texts, raw_emotions, sarcasm_probs):
        is_sarcastic = s_prob >= sarcasm_threshold
        final = emos
        if is_sarcastic and emos and emos[0][0] in POSITIVE_VALENCE:
            final = [(FLIP_TO, s_prob)] + emos[1:]
        results.append(FusedResult(text, emos, s_prob, is_sarcastic, final))
    return results

if __name__ == "__main__":
    from inference import predict as emotion_predict
    from sarcasm_model import predict_sarcasm_proba

    samples = [
        "This is exactly what I needed today, thank you so much!",
        "Oh great, another power cut. Just what I needed.",
        "Sure, love that the train is delayed again.",
    ]
    for r in fuse(samples, emotion_predict, predict_sarcasm_proba):
        print(f"{r.text!r}")
        print(f"  raw: {r.raw_emotions}  sarcasm_prob={r.sarcasm_prob:.2f} "
              f"sarcastic={r.sarcasm_detected}")
        print(f"  final: {r.final_emotions}\n")
