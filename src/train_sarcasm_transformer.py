import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)
from sklearn.metrics import f1_score, accuracy_score

try:
    from .sarcasm_preprocess import train_test
except ImportError:
    from src.sarcasm_preprocess import train_test

DEFAULT_OUT_DIR = str(Path(__file__).resolve().parents[1] / "models" / "sarcasm_transformer")

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.enc = tokenizer(list(texts), truncation=True, padding="max_length",
                              max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base",
                     choices=["roberta-base", "distilbert-base-uncased"])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    train_df, test_df = train_test()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_ds = SarcasmDataset(train_df["text"], train_df["label"], tokenizer)
    test_ds = SarcasmDataset(test_df["text"], test_df["label"], tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved fine-tuned sarcasm model to {args.out_dir}/")

if __name__ == "__main__":
    main()