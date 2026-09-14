import time
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report

from preprocess import load_split, EMOTIONS

def main():
    print("Loading data...")
    train_df = load_split("train")
    dev_df = load_split("dev")

    mlb = MultiLabelBinarizer(classes=list(range(len(EMOTIONS))))
    y_train = mlb.fit_transform(train_df["labels"])
    y_dev = mlb.transform(dev_df["labels"])

    print("Vectorizing text (TF-IDF, word 1-2 grams)...")
    vectorizer = TfidfVectorizer(
        max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(train_df["text"])
    X_dev = vectorizer.transform(dev_df["text"])

    print(f"Training OvR Logistic Regression on {X_train.shape[0]} examples, "
          f"{X_train.shape[1]} features, {len(EMOTIONS)} labels...")
    t0 = time.time()
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=200, class_weight="balanced", C=1.0),
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print(f"Trained in {time.time() - t0:.1f}s")

    print("\nEvaluating on dev set...")
    y_pred = clf.predict(X_dev)
    micro_f1 = f1_score(y_dev, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_dev, y_pred, average="macro", zero_division=0)
    print(f"Dev micro-F1: {micro_f1:.4f}  |  Dev macro-F1: {macro_f1:.4f}")
    print("\nPer-emotion report:")
    print(classification_report(y_dev, y_pred, target_names=EMOTIONS, zero_division=0))

    joblib.dump({"vectorizer": vectorizer, "clf": clf, "labels": EMOTIONS},
                "emotion_baseline.joblib")
    print("Saved model -> emotion_baseline.joblib")

if __name__ == "__main__":
    main()
