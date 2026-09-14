import time
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report

from sarcasm_preprocess import train_test

def main():
    train_df, test_df = train_test()

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    print(f"Training on {X_train.shape[0]} examples...")
    t0 = time.time()
    clf = LogisticRegression(max_iter=300, C=1.0)
    clf.fit(X_train, train_df["label"])
    print(f"Trained in {time.time() - t0:.1f}s")

    preds = clf.predict(X_test)
    acc = accuracy_score(test_df["label"], preds)
    f1 = f1_score(test_df["label"], preds)
    print(f"\nTest accuracy: {acc:.4f}  |  Test F1 (sarc): {f1:.4f}")
    print(classification_report(test_df["label"], preds, target_names=["notsarc", "sarc"]))

    joblib.dump({"vectorizer": vectorizer, "clf": clf}, "sarcasm_baseline.joblib")
    print("Saved model -> sarcasm_baseline.joblib")

if __name__ == "__main__":
    main()
