
"""Fine-tune the best continued DistilBERT checkpoint in an isolated experiment."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
import hashlib
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from train_baseline import prepare_data
from model_runtime import LABELS


SOURCE = ROOT / "artifacts" / "distilbert_continued" / "best"
OUTPUT = ROOT / "artifacts" / "distilbert_experiment_2"
BEST = OUTPUT / "best_model"


class EncodedDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.encodings = tokenizer(
            [r[0] for r in rows], truncation=True, max_length=96
        )
        self.labels = [LABELS.index(r[1]) for r in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            **{k: v[index] for k, v in self.encodings.items()},
            "labels": self.labels[index],
        }


class LengthGroupedBatches:
    def __init__(self, dataset, batch_size, shuffle):
        self.indices = sorted(
            range(len(dataset)),
            key=lambda i: len(dataset.encodings["input_ids"][i])
        )
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batches = [
            self.indices[i:i + self.batch_size]
            for i in range(0, len(self.indices), self.batch_size)
        ]
        if self.shuffle:
            random.shuffle(batches)
        yield from batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--trainable-layers", type=int, choices=range(1, 7), default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("Epochs, batch size and learning rate must be positive.")

    if not SOURCE.exists():
        raise FileNotFoundError(f"Starting checkpoint not found: {SOURCE}")

    if OUTPUT.exists():
        raise RuntimeError(
            f"{OUTPUT} already exists. Nothing was overwritten. "
            "Choose a new output folder before running again."
        )

    OUTPUT.mkdir(parents=True)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading existing model from: {SOURCE}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(SOURCE), local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(SOURCE), local_files_only=True
    )

    for parameter in model.distilbert.parameters():
        parameter.requires_grad = False

    for layer in model.distilbert.transformer.layer[-args.trainable_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    model.to(device)

    splits, audit = prepare_data()

    def collate(rows):
        targets = torch.tensor([row["labels"] for row in rows])
        inputs = tokenizer.pad(
            [{k: v for k, v in row.items() if k != "labels"} for row in rows],
            return_tensors="pt"
        )
        return inputs, targets

    datasets = {
        name: EncodedDataset(rows, tokenizer)
        for name, rows in splits.items()
    }

    loaders = {
        name: (
            DataLoader(
                data,
                batch_sampler=LengthGroupedBatches(data, args.batch_size, True),
                collate_fn=collate,
                num_workers=0
            )
            if name == "train"
            else DataLoader(
                data,
                batch_size=args.batch_size,
                collate_fn=collate,
                num_workers=0
            )
        )
        for name, data in datasets.items()
    }

    def predict(loader):
        model.eval()
        probabilities = []
        with torch.inference_mode():
            for inputs, _ in loader:
                outputs = model(
                    **{k: v.to(device) for k, v in inputs.items()}
                )
                probabilities.append(
                    outputs.logits.softmax(-1).cpu().numpy()
                )
        return np.concatenate(probabilities)

    # Preserve the starting checkpoint as the initial best model.
    initial_probs = predict(loaders["val"])
    best_score = f1_score(
        datasets["val"].labels,
        initial_probs.argmax(1),
        average="macro"
    )
    best_epoch = 0

    model.save_pretrained(BEST, safe_serialization=True)
    tokenizer.save_pretrained(BEST)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01
    )

    total_steps = args.epochs * len(loaders["train"])
    warmup_steps = max(1, int(0.1 * total_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(
            (step + 1) / warmup_steps,
            max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
        )
    )

    history = []
    started = time.perf_counter()

    print(
        f"Device={device}; starting validation macro-F1={best_score:.4f}; "
        f"epochs={args.epochs}; trainable layers={args.trainable_layers}",
        flush=True
    )

    for epoch in range(args.epochs):
        model.train()
        model.distilbert.embeddings.eval()

        for layer in model.distilbert.transformer.layer[:-args.trainable_layers]:
            layer.eval()

        running_loss = 0.0

        for step, (inputs, targets) in enumerate(loaders["train"]):
            optimizer.zero_grad(set_to_none=True)

            logits = model(
                **{k: v.to(device) for k, v in inputs.items()}
            ).logits

            loss = torch.nn.functional.cross_entropy(
                logits, targets.to(device)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            if step % 25 == 0:
                progress = {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "steps_per_epoch": len(loaders["train"]),
                    "loss": running_loss / (step + 1),
                    "best_validation_macro_f1": best_score,
                    "elapsed_seconds": round(time.perf_counter() - started)
                }
                (OUTPUT / "progress.json").write_text(
                    json.dumps(progress, indent=2), encoding="utf-8"
                )
                print(json.dumps(progress), flush=True)

        validation_probs = predict(loaders["val"])
        score = f1_score(
            datasets["val"].labels,
            validation_probs.argmax(1),
            average="macro"
        )

        record = {
            "epoch": epoch + 1,
            "validation_macro_f1": float(score),
            "training_loss": running_loss / len(loaders["train"])
        }
        history.append(record)

        print(json.dumps(record), flush=True)

        if score > best_score:
            best_score = float(score)
            best_epoch = epoch + 1
            model.save_pretrained(BEST, safe_serialization=True)
            tokenizer.save_pretrained(BEST)
            print("New best validation checkpoint saved.", flush=True)

        (OUTPUT / "training_history.json").write_text(
            json.dumps({
                "configuration": vars(args),
                "starting_validation_macro_f1": float(
                    f1_score(
                        datasets["val"].labels,
                        initial_probs.argmax(1),
                        average="macro"
                    )
                ),
                "best_validation_macro_f1": best_score,
                "best_epoch": best_epoch,
                "history": history
            }, indent=2),
            encoding="utf-8"
        )

    # Evaluate the best validation checkpoint, not necessarily the final epoch.
    del model
    model = AutoModelForSequenceClassification.from_pretrained(
        str(BEST), local_files_only=True
    ).to(device)

    def final_predict(loader):
        model.eval()
        probabilities = []
        with torch.inference_mode():
            for inputs, _ in loader:
                logits = model(
                    **{k: v.to(device) for k, v in inputs.items()}
                ).logits
                probabilities.append(logits.softmax(-1).cpu().numpy())
        return np.concatenate(probabilities)

    validation_probs = final_predict(loaders["val"])
    threshold_trials = []

    for threshold in np.arange(0.35, 0.86, 0.05):
        keep = validation_probs.max(1) >= threshold
        threshold_trials.append({
            "threshold": round(float(threshold), 2),
            "coverage": float(keep.mean()),
            "accuracy": float(
                (validation_probs.argmax(1)[keep] ==
                 np.array(datasets["val"].labels)[keep]).mean()
            ) if keep.any() else 0.0
        })

    eligible = [
        r for r in threshold_trials
        if r["accuracy"] >= 0.85 and r["coverage"] >= 0.3
    ]
    threshold = (
        max(eligible, key=lambda r: r["coverage"])["threshold"]
        if eligible else 0.5
    )

    # Test set is evaluated only after model selection is complete.
    test_probs = final_predict(loaders["test"])
    labels = datasets["test"].labels
    predicted = test_probs.argmax(1)

    report = {
        "model_version": "distilbert-experiment-2",
        "starting_checkpoint": str(SOURCE),
        "configuration": vars(args),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "data_audit": audit,
        "validation_history": history,
        "threshold_trials": threshold_trials,
        "uncertainty_threshold": threshold,
        "training_seconds": time.perf_counter() - started,
        "test": {
            "accuracy": accuracy_score(labels, predicted),
            "macro_f1": f1_score(labels, predicted, average="macro"),
            "classification_report": classification_report(
                labels, predicted,
                labels=list(range(6)),
                target_names=LABELS,
                output_dict=True,
                zero_division=0
            ),
            "labels": LABELS,
            "confusion_matrix": confusion_matrix(
                labels, predicted, labels=list(range(6))
            ).tolist()
        }
    }

    weights_path = BEST / "model.safetensors"
    metadata = {
        "version": "distilbert-experiment-2",
        "threshold": threshold,
        "weights_sha256": hashlib.sha256(
            weights_path.read_bytes()
        ).hexdigest()
    }

    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\nEXPERIMENT COMPLETE", flush=True)
    print(json.dumps(report["test"], indent=2), flush=True)
    print(f"Best validation macro-F1: {best_score:.4f}", flush=True)
    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Results saved in: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
