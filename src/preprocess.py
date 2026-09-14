import pandas as pd
from pathlib import Path

# Path relative to project root
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

with open(DATA_DIR / "emotions.txt") as f:
    EMOTIONS = [l.strip() for l in f if l.strip()]  # 28 labels, index-aligned

def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{name}.tsv", sep="\t", header=None,
                      names=["text", "label_ids", "rater_id"])
    # label_ids is a comma-separated string of indices into EMOTIONS, e.g. "2,7"
    df["labels"] = df["label_ids"].apply(lambda s: [int(x) for x in str(s).split(",")])
    return df[["text", "labels"]]

if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        df = load_split(split)
        print(f"{split}: {len(df)} rows, e.g. {df.iloc[0]['text'][:50]!r} -> "
              f"{[EMOTIONS[i] for i in df.iloc[0]['labels']]}")
