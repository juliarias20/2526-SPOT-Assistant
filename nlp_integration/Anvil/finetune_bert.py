"""
finetune_bert.py
-----------------
Fine-tunes bert-base-uncased as a 6-class intent classifier on the SPOT
robot command dataset. Designed to run on Purdue Anvil GPU nodes (A100/H100).

After training, model is saved locally so it can be loaded offline —
no HuggingFace download needed at inference time.

Usage (on Anvil, inside Slurm job or interactive session):
    python finetune_bert.py \
        --data    training_data.jsonl \
        --labels  label_map.json \
        --out     ./models/bert-spot-intent \
        --epochs  10 \
        --batch   32

Requirements:
    pip install transformers datasets scikit-learn torch accelerate
"""

import argparse
import json
import os
import random
import numpy as np
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings.get("token_type_ids", torch.zeros_like(self.encodings["input_ids"]))[idx],
            "labels":         self.labels[idx],
        }


def load_data(data_path: str):
    texts, labels = [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            texts.append(rec["text"])
            labels.append(rec["label"])
    print(f"[INFO] Loaded {len(texts)} records from {data_path}")
    dist = Counter(labels)
    print(f"[INFO] Label distribution: {dict(sorted(dist.items()))}")
    return texts, labels


# ─────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load label map ────────────────────────────────────────────
    with open(args.labels, "r") as f:
        label_map: dict = json.load(f)  # {"intent_name": int_id, ...}
    id2label = {v: k for k, v in label_map.items()}
    num_labels = len(label_map)
    print(f"[INFO] {num_labels} intent classes: {list(label_map.keys())}")

    # ── Load data ─────────────────────────────────────────────────
    texts, labels = load_data(args.data)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=args.val_split,
        stratify=labels,
        random_state=args.seed,
    )
    print(f"[INFO] Train: {len(train_texts)} | Val: {len(val_texts)}")

    # ── Tokenizer & model ─────────────────────────────────────────
    print(f"[INFO] Loading tokenizer: {args.base_model}")
    tokenizer = BertTokenizerFast.from_pretrained(args.base_model)

    print(f"[INFO] Loading model: {args.base_model}")
    model = BertForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label_map,
    )
    model.to(device)

    # ── Datasets & loaders ────────────────────────────────────────
    train_dataset = IntentDataset(train_texts, train_labels, tokenizer, max_len=args.max_len)
    val_dataset   = IntentDataset(val_texts,   val_labels,   tokenizer, max_len=args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    # ── Optimizer & scheduler ─────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training ──────────────────────────────────────────────────
    best_val_acc = 0.0
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Starting training for {args.epochs} epochs...")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        # — Train —
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # — Validate —
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_true.extend(batch["labels"].cpu().numpy().tolist())

        val_acc = sum(p == t for p, t in zip(all_preds, all_true)) / len(all_true)
        print(f"Epoch {epoch:02d}/{args.epochs:02d}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(str(out_path))
            tokenizer.save_pretrained(str(out_path))
            print(f"  ✓ New best ({val_acc:.4f}) — saved to {out_path}")

    print("=" * 60)
    print(f"\n[INFO] Training complete. Best val accuracy: {best_val_acc:.4f}")

    # ── Final evaluation on val set with best model ───────────────
    print("\n[INFO] Loading best checkpoint for final evaluation...")
    model = BertForSequenceClassification.from_pretrained(str(out_path))
    model.to(device)
    model.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_true.extend(batch["labels"].cpu().numpy().tolist())

    target_names = [id2label[i] for i in range(num_labels)]
    print("\n[RESULTS] Classification Report:")
    print(classification_report(all_true, all_preds, target_names=target_names))

    print("[RESULTS] Confusion Matrix:")
    cm = confusion_matrix(all_true, all_preds)
    print(cm)

    # Save label map alongside model for easy loading
    with open(out_path / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\n[OK] Model, tokenizer, and label_map.json saved → {out_path}")
    print("[OK] Transfer this directory to your local machine to use offline.")


# ─────────────────────────────────────────────────────────────────
# QUICK SMOKE TEST  (run after fine-tuning to verify offline load)
# ─────────────────────────────────────────────────────────────────

def smoke_test(model_path: str):
    print(f"\n[TEST] Loading model from {model_path} ...")
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    with open(Path(model_path) / "label_map.json") as f:
        label_map = json.load(f)
    id2label = {v: k for k, v in label_map.items()}

    test_cmds = [
        ("Get me a pen.",                                  "retrieve_object"),
        ("Find my phone.",                                 "locate_object"),
        ("Scan the room.",                                 "scan_environment"),
        ("Go to the kitchen.",                             "navigate"),
        ("Go to the desk and bring me the notebook.",      "multi_step_retrieve"),
        ("Pick up the bottle and put it on the shelf.",    "multi_step_manipulation"),
    ]

    print("\n{:<50} {:<30} {:<30} {}".format("Command", "Expected", "Predicted", "✓"))
    print("-" * 130)
    correct = 0
    for cmd, expected in test_cmds:
        inputs = tokenizer(cmd, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()
        predicted = id2label[pred_id]
        ok = "✓" if predicted == expected else "✗"
        if predicted == expected:
            correct += 1
        print(f"  {cmd:<48} {expected:<30} {predicted:<30} {ok}")
    print(f"\nSmoke test accuracy: {correct}/{len(test_cmds)}")


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for SPOT intent classification")
    parser.add_argument("--data",       default="training_data.jsonl", help="Training JSONL")
    parser.add_argument("--labels",     default="label_map.json",      help="Label map JSON")
    parser.add_argument("--out",        default="./models/bert-spot-intent", help="Output directory")
    parser.add_argument("--base-model", default="bert-base-uncased",   help="HuggingFace model ID")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch",      type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max-len",    type=int,   default=128)
    parser.add_argument("--val-split",  type=float, default=0.15)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run smoke test on saved model and exit (skip training)")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test(args.out)
    else:
        train(args)


if __name__ == "__main__":
    main()
