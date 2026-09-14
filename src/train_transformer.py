import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

try:
    from .preprocess import load_split, EMOTIONS
except ImportError:
    from src.preprocess import load_split, EMOTIONS

DEFAULT_OUT_DIR = str(Path(__file__).resolve().parents[1] / "models" / "emotion_transformer")

class GoEmotionsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.enc = tokenizer(list(texts), truncation=True, padding="max_length",
                              max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.3).astype(int)
    return {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base",
                     choices=["roberta-base", "distilbert-base-uncased"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    train_df = load_split("train")
    dev_df = load_split("dev")

    mlb = MultiLabelBinarizer(classes=list(range(len(EMOTIONS))))
    y_train = mlb.fit_transform(train_df["labels"])
    y_dev = mlb.transform(dev_df["labels"])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_ds = GoEmotionsDataset(train_df["text"], y_train, tokenizer)
    dev_ds = GoEmotionsDataset(dev_df["text"], y_dev, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(EMOTIONS), problem_type="multi_label_classification"
    )

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        logging_steps=100,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved fine-tuned model to {args.out_dir}/")

if __name__ == "__main__":
    main()
