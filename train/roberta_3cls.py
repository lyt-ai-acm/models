# -*- coding: utf-8 -*-
"""
Weibo sentiment three-class training script (adapted from the binary classification script)
Supports two data formats:
1) Data already has three-class labels (e.g., 0/1/2 or neg/neu/pos)
2) Data is binary (0/1) and you provide a neutral data file to merge

Core capabilities:
- Label mapping (automatic or manual)
- Optional class weight loss (CrossEntropy + class_weight)
- Evaluation metrics: Accuracy / Macro-F1 / per-class P-R-F1 / confusion matrix
- Save best model (by macro_f1)

Example A (already three-class):
python train/train_roberta_3cls.py ^
  --data_path data/weibo_3cls.csv ^
  --text_col review --label_col label ^
  --output_dir outputs/roberta_3cls ^
  --model_name hfl/chinese-roberta-wwm-ext ^
  --epochs 3 --batch_size 16 --lr 2e-5 --max_len 128 --seed 42

Example B (binary + neutral set merge):
python train/train_roberta_3cls.py ^
  --data_path data/Weibo_senti_100k.csv ^
  --neutral_data_path data/weibo_neutral.csv ^
  --text_col review --label_col label ^
  --neutral_text_col review ^
  --output_dir outputs/roberta_3cls ^
  --epochs 3 --batch_size 16
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


# -----------------------------
# Arguments
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True, help="Main data CSV")
    p.add_argument("--neutral_data_path", type=str, default="", help="Optional: neutral data CSV (for extending binary to three-class)")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_name", type=str, default="hfl/chinese-roberta-wwm-ext")

    p.add_argument("--text_col", type=str, default="review")
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--neutral_text_col", type=str, default="review")

    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)

    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--dev_size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--logging_steps", type=int, default=10)
    # Label mapping strategy
    p.add_argument("--label_mode", type=str, default="auto", choices=["auto", "manual"],
                   help="auto automatically identifies labels; manual uses the manual mapping below")
    p.add_argument("--manual_neg_values", type=str, default="0,neg,negative,负面")
    p.add_argument("--manual_neu_values", type=str, default="1,neu,neutral,中性")
    p.add_argument("--manual_pos_values", type=str, default="2,pos,positive,正面")

    # Loss function
    p.add_argument("--use_class_weight", action="store_true", help="Whether to use class weights")
    return p.parse_args()


# -----------------------------
# Label utilities
# -----------------------------
def _norm_label_value(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    return s


def parse_value_set(s: str) -> set:
    return set([x.strip().lower() for x in s.split(",") if x.strip()])


def map_labels_auto(raw_labels: List[Any]) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Automatic mapping logic:
    - If only {0,1,2} -> directly map to the same values
    - If only {0,1} -> default 0=neg, 1=pos (no neu)
    - If text labels, recognize neg/neu/pos synonyms
    """
    vals = [_norm_label_value(x) for x in raw_labels]
    uniq = sorted(set(vals))

    neg_alias = {"0", "neg", "negative", "负面", "-1"}
    neu_alias = {"1", "neu", "neutral", "中性"}
    pos_alias = {"2", "pos", "positive", "正面"}

    # case1: 0/1/2
    if set(uniq).issubset({"0", "1", "2"}):
        y = np.array([int(v) for v in vals], dtype=np.int64)
        id2label = {0: "neg", 1: "neu", 2: "pos"}
        label2id = {"neg": 0, "neu": 1, "pos": 2}
        return y, label2id, id2label

    # case2: text mapping
    y = []
    for v in vals:
        if v in neg_alias:
            y.append(0)
        elif v in neu_alias:
            y.append(1)
        elif v in pos_alias:
            y.append(2)
        else:
            raise ValueError(f"auto mode cannot recognize label value: {v}")
    y = np.array(y, dtype=np.int64)
    id2label = {0: "neg", 1: "neu", 2: "pos"}
    label2id = {"neg": 0, "neu": 1, "pos": 2}
    return y, label2id, id2label


def map_labels_manual(raw_labels: List[Any], neg_set: set, neu_set: set, pos_set: set) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    vals = [_norm_label_value(x) for x in raw_labels]
    y = []
    for v in vals:
        if v in neg_set:
            y.append(0)
        elif v in neu_set:
            y.append(1)
        elif v in pos_set:
            y.append(2)
        else:
            raise ValueError(f"manual mode did not match label value: {v}")
    y = np.array(y, dtype=np.int64)
    id2label = {0: "neg", 1: "neu", 2: "pos"}
    label2id = {"neg": 0, "neu": 1, "pos": 2}
    return y, label2id, id2label


# -----------------------------
# Custom Trainer: supports class weight
# -----------------------------
class WeightedCELossTrainer(Trainer):
    def __init__(self, class_weights: Optional[torch.Tensor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 3), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# -----------------------------
# Metrics
# -----------------------------
def build_compute_metrics(id2label: Dict[int, str]):
    def _fn(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")

        p, r, f1, support = precision_recall_fscore_support(labels, preds, labels=[0, 1, 2], zero_division=0)
        cm = confusion_matrix(labels, preds, labels=[0, 1, 2])

        out = {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "neg_precision": float(p[0]), "neg_recall": float(r[0]), "neg_f1": float(f1[0]), "neg_support": int(support[0]),
            "neu_precision": float(p[1]), "neu_recall": float(r[1]), "neu_f1": float(f1[1]), "neu_support": int(support[1]),
            "pos_precision": float(p[2]), "pos_recall": float(r[2]), "pos_f1": float(f1[2]), "pos_support": int(support[2]),
            "cm_00": int(cm[0, 0]), "cm_01": int(cm[0, 1]), "cm_02": int(cm[0, 2]),
            "cm_10": int(cm[1, 0]), "cm_11": int(cm[1, 1]), "cm_12": int(cm[1, 2]),
            "cm_20": int(cm[2, 0]), "cm_21": int(cm[2, 1]), "cm_22": int(cm[2, 2]),
        }
        return out
    return _fn


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # 1) Read main data
    df = pd.read_csv(args.data_path, encoding="utf-8-sig")
    assert args.text_col in df.columns, f"Missing text column: {args.text_col}"
    assert args.label_col in df.columns, f"Missing label column: {args.label_col}"

    df = df[[args.text_col, args.label_col]].copy()
    df[args.text_col] = df[args.text_col].fillna("").astype(str).str.strip()
    df = df[df[args.text_col] != ""].copy()

    # 2) Optionally merge neutral data (extend binary to three-class)
    # Convention: all neutral data labeled as 1 (neu)
    if args.neutral_data_path:
        neu_df = pd.read_csv(args.neutral_data_path, encoding="utf-8-sig")
        assert args.neutral_text_col in neu_df.columns, f"Neutral data missing text column: {args.neutral_text_col}"
        neu_df = neu_df[[args.neutral_text_col]].rename(columns={args.neutral_text_col: args.text_col})
        neu_df[args.text_col] = neu_df[args.text_col].fillna("").astype(str).str.strip()
        neu_df = neu_df[neu_df[args.text_col] != ""].copy()
        neu_df[args.label_col] = 1  # neu
        df = pd.concat([df, neu_df], axis=0, ignore_index=True)

    # 3) Label mapping
    raw_labels = df[args.label_col].tolist()
    if args.label_mode == "auto":
        y, label2id, id2label = map_labels_auto(raw_labels)
    else:
        neg_set = parse_value_set(args.manual_neg_values)
        neu_set = parse_value_set(args.manual_neu_values)
        pos_set = parse_value_set(args.manual_pos_values)
        y, label2id, id2label = map_labels_manual(raw_labels, neg_set, neu_set, pos_set)

    # If binary data (only 0/1 and auto-mapped to neg/neu), pos is missing, training three-class is unreasonable
    uniq = sorted(set(y.tolist()))
    if len(uniq) < 3:
        raise ValueError(
            f"Current data contains only {uniq} classes, not reaching three classes. "
            f"Please provide neutral_data_path or use true three-class data."
        )

    df["label_id"] = y

    # 4) Split (stratified)
    train_dev, test = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, stratify=df["label_id"]
    )
    dev_ratio = args.dev_size / (1.0 - args.test_size)
    train, dev = train_test_split(
        train_dev, test_size=dev_ratio, random_state=args.seed, stratify=train_dev["label_id"]
    )

    train.to_csv(os.path.join(args.output_dir, "train.csv"), index=False, encoding="utf-8-sig")
    dev.to_csv(os.path.join(args.output_dir, "dev.csv"), index=False, encoding="utf-8-sig")
    test.to_csv(os.path.join(args.output_dir, "test.csv"), index=False, encoding="utf-8-sig")

    print(f"[Split] train={len(train)}, dev={len(dev)}, test={len(test)}")
    print("[Train label dist]")
    print(train["label_id"].value_counts(normalize=True).sort_index())

    # 5) datasets
    def to_hf(d):
        return Dataset.from_pandas(
            d[[args.text_col, "label_id"]].rename(columns={args.text_col: "text", "label_id": "labels"}),
            preserve_index=False
        )

    ds = DatasetDict({
        "train": to_hf(train),
        "dev": to_hf(dev),
        "test": to_hf(test),
    })

    # 6) tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name)

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tok)

    # 7) model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )

    # 8) Class weights
    class_weights = None
    if args.use_class_weight:
        cls = np.array([0, 1, 2], dtype=np.int64)
        w = compute_class_weight(class_weight="balanced", classes=cls, y=train["label_id"].values)
        class_weights = torch.tensor(w, dtype=torch.float32)
        print("[ClassWeights]", w)

    # 9) trainer
    targs = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        disable_tqdm=False,

        fp16=args.fp16,
        report_to="none",
        seed=args.seed,
    )

    trainer = WeightedCELossTrainer(
        model=model,
        args=targs,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["dev"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=build_compute_metrics(id2label),
        class_weights=class_weights,
    )

    trainer.train()

    # 10) eval
    dev_metrics = trainer.evaluate(ds_tok["dev"])
    test_metrics = trainer.evaluate(ds_tok["test"])

    # Manually compute a full confusion matrix
    pred = trainer.predict(ds_tok["test"])
    y_true = pred.label_ids
    y_pred = np.argmax(pred.predictions, axis=-1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    # 11) save
    best_dir = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_dir)
    tok.save_pretrained(best_dir)

    metrics = {
        "dev": {k: float(v) for k, v in dev_metrics.items() if isinstance(v, (int, float))},
        "test": {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))},
        "test_confusion_matrix": cm,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "config": vars(args),
    }
    with open(os.path.join(args.output_dir, "metrics_3cls.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[Done] saved:", args.output_dir)


if __name__ == "__main__":
    main()
