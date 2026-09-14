import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

from preprocess import load_split, EMOTIONS


class GoEmotionsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.enc = tokenizer(list(texts), truncation=True, padding="max_length",
                              max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """Per-class weight for BCEWithLogitsLoss: (negatives / positives) for
    each of the 28 emotions, capped to avoid instability on ultra-rare
    classes. Up-weights rare emotions so the model doesn't just learn to
    always predict 'neutral' / the most common labels."""
    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts
    weights = neg_counts / np.clip(pos_counts, 1, None)
    return torch.tensor(np.clip(weights, 1.0, 20.0), dtype=torch.float)


class WeightedTrainer(Trainer):
    """Trainer with class-weighted BCE loss instead of the default
    unweighted BCE that HF uses for problem_type='multi_label_classification'."""

    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Metrics during training use a fixed 0.3 threshold for a quick,
    consistent signal across epochs; the real per-label thresholds are
    tuned separately after training in tune_thresholds()."""
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.3).astype(int)
    return {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def tune_thresholds(probs: np.ndarray, labels: np.ndarray, grid=None):
    """Finds the per-class threshold in `grid` that maximizes that class's
    F1 on the dev set. Returns a (28,) array of thresholds."""
    if grid is None:
        grid = np.arange(0.05, 0.95, 0.05)
    best_thresholds = np.full(labels.shape[1], 0.5)
    for i in range(labels.shape[1]):
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            preds = (probs[:, i] >= t).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[i] = best_t
    return best_thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base",
                     choices=["roberta-base", "distilbert-base-uncased"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--patience", type=int, default=2,
                     help="Early-stopping patience, in eval epochs")
    ap.add_argument("--out_dir", default="emotion_transformer")
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
    pos_weight = compute_pos_weight(y_train)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
        pos_weight=pos_weight,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()

    # Tune per-label thresholds on dev set using the best checkpoint
    print("\nTuning per-label decision thresholds on dev set...")
    dev_logits = trainer.predict(dev_ds).predictions
    dev_probs = 1 / (1 + np.exp(-dev_logits))
    thresholds = tune_thresholds(dev_probs, y_dev)

    tuned_preds = (dev_probs >= thresholds).astype(int)
    print(f"Dev micro-F1 (tuned thresholds): "
          f"{f1_score(y_dev, tuned_preds, average='micro', zero_division=0):.4f}")
    print(f"Dev macro-F1 (tuned thresholds): "
          f"{f1_score(y_dev, tuned_preds, average='macro', zero_division=0):.4f}  "
          f"(vs fixed 0.3 threshold — see per-epoch logs above)")

    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    with open(f"{args.out_dir}/thresholds.json", "w") as f:
        json.dump({EMOTIONS[i]: float(thresholds[i]) for i in range(len(EMOTIONS))}, f, indent=2)
    print(f"Saved fine-tuned model + per-label thresholds to {args.out_dir}/")
    print("NOTE: inference.py currently uses a single fixed threshold. To "
          "benefit from the tuned per-label thresholds at inference time, "
          "load thresholds.json there and compare each label's probability "
          "against its own threshold instead of one global value.")


if __name__ == "__main__":
    main()