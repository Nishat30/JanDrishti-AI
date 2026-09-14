import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent / "sarcasm_data"
SUBSETS = ["GEN-sarc-notsarc.csv", "HYP-sarc-notsarc.csv", "RQ-sarc-notsarc.csv"]

def load_all(dedupe=True) -> pd.DataFrame:
    frames = []
    for f in SUBSETS:
        df = pd.read_csv(DATA_DIR / f)
        df["subset"] = f.split("-")[0]
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full["label"] = (full["class"] == "sarc").astype(int)
    if dedupe:
        # HYP/RQ posts overlap with GEN by design (per the corpus README)
        full = full.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return full[["text", "label", "subset"]]

def train_test(test_size=0.15, seed=42):
    df = load_all()
    return train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)

if __name__ == "__main__":
    df = load_all()
    print(f"Total: {len(df)} rows, sarc={df['label'].sum()}, notsarc={(df['label']==0).sum()}")
    print(df["subset"].value_counts())
