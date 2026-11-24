"""Fine-tune DistilBERT on exactly the baseline's cleaned splits, locally."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / 'artifacts' / 'hf_cache'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
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


class EncodedDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.encodings = tokenizer([r[0] for r in rows], truncation=True, max_length=96)
        self.labels = [LABELS.index(r[1]) for r in rows]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return {**{k: v[index] for k, v in self.encodings.items()}, 'labels': self.labels[index]}


class LengthGroupedBatches:
    """Shuffle batch order while avoiding padding short examples to long outliers."""
    def __init__(self, dataset, batch_size, shuffle):
        self.indices = sorted(range(len(dataset)), key=lambda i: len(dataset.encodings['input_ids'][i]))
        self.batch_size, self.shuffle = batch_size, shuffle
    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size
    def __iter__(self):
        batches = [self.indices[i:i + self.batch_size] for i in range(0, len(self.indices), self.batch_size)]
        if self.shuffle:
            random.shuffle(batches)
        yield from batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--trainable-layers', type=int, choices=range(1, 7), default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch size must be positive')
    torch.set_num_threads(args.threads)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = ROOT / 'artifacts' / 'distilbert'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'metadata.json').exists():
        raise RuntimeError('A completed model already exists. Preserve it in a separate experiment folder before retraining.')
    checkpoint = 'distilbert/distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=6, id2label=dict(enumerate(LABELS)), label2id={v: i for i, v in enumerate(LABELS)}, use_safetensors=True)
    checkpoint_revision = getattr(model.config, '_commit_hash', None)
    for parameter in model.distilbert.parameters():
        parameter.requires_grad = False
    for layer in model.distilbert.transformer.layer[-args.trainable_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    model.to(device)
    splits, audit = prepare_data()
    def collate(rows):
        targets = torch.tensor([row['labels'] for row in rows])
        inputs = tokenizer.pad([{k: v for k, v in row.items() if k != 'labels'} for row in rows], return_tensors='pt')
        return inputs, targets
    datasets = {name: EncodedDataset(rows, tokenizer) for name, rows in splits.items()}
    loaders = {name: (DataLoader(data, batch_sampler=LengthGroupedBatches(data, args.batch_size, True), collate_fn=collate, num_workers=0) if name == 'train' else DataLoader(data, batch_size=args.batch_size, collate_fn=collate, num_workers=0)) for name, data in datasets.items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5, weight_decay=.01)
    total_steps = args.epochs * len(loaders['train'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / max(1, int(.1 * total_steps)), max(0., (total_steps - step) / max(1, .9 * total_steps))))
    def predict(loader):
        model.eval()
        output_probs = []
        with torch.inference_mode():
            for inputs, _ in loader:
                output_probs.append(model(**{k: v.to(device) for k, v in inputs.items()}).logits.softmax(-1).cpu().numpy())
        return np.concatenate(output_probs)
    started = time.perf_counter()
    best, history = -1., []
    print(f'Device={device}; training examples={len(datasets["train"])}; trainable last layers={args.trainable_layers}; steps={total_steps}', flush=True)
    for epoch in range(args.epochs):
        model.train()
        # Frozen encoder blocks are deterministic during partial fine-tuning.
        model.distilbert.embeddings.eval()
        for layer in model.distilbert.transformer.layer[:-args.trainable_layers]:
            layer.eval()
        running_loss = 0.
        for step, (inputs, targets) in enumerate(loaders['train']):
            optimizer.zero_grad(set_to_none=True)
            logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits
            loss = torch.nn.functional.cross_entropy(logits, targets.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            if step % 25 == 0:
                status = {'epoch': epoch + 1, 'step': step + 1, 'steps_per_epoch': len(loaders['train']), 'loss': running_loss / (step + 1), 'elapsed_seconds': round(time.perf_counter() - started)}
                (output / 'progress.json').write_text(json.dumps(status), encoding='utf-8')
                print(json.dumps(status), flush=True)
        probs = predict(loaders['val'])
        score = f1_score(datasets['val'].labels, probs.argmax(1), average='macro')
        history.append({'epoch': epoch + 1, 'validation_macro_f1': score, 'training_loss': running_loss / len(loaders['train'])})
        (output / 'training_history.json').write_text(json.dumps({'configuration': vars(args), 'history': history}, indent=2), encoding='utf-8')
        print(json.dumps(history[-1]), flush=True)
        if score > best:
            best = score
            model.save_pretrained(output, safe_serialization=True)
            tokenizer.save_pretrained(output)
            (output / 'checkpoint_state.json').write_text(json.dumps({'saved_epoch': epoch + 1, 'validation_macro_f1': score, 'checkpoint_revision': checkpoint_revision}, indent=2), encoding='utf-8')
    model = AutoModelForSequenceClassification.from_pretrained(output, local_files_only=True).to(device)
    validation_probs = predict(loaders['val'])
    threshold_trials = []
    for threshold in np.arange(.35, .86, .05):
        keep = validation_probs.max(1) >= threshold
        threshold_trials.append({'threshold': round(float(threshold), 2), 'coverage': float(keep.mean()), 'accuracy': float((validation_probs.argmax(1)[keep] == np.array(datasets['val'].labels)[keep]).mean()) if keep.any() else 0.})
    eligible = [r for r in threshold_trials if r['accuracy'] >= .85 and r['coverage'] >= .3]
    threshold = max(eligible, key=lambda r: r['coverage'])['threshold'] if eligible else .5
    probs = predict(loaders['test'])
    labels, predicted = datasets['test'].labels, probs.argmax(1)
    version = 'distilbert-partial-v1'
    report = {'model_version': version, 'checkpoint': checkpoint, 'checkpoint_revision': checkpoint_revision, 'selected_epoch': max(history, key=lambda row: row['validation_macro_f1'])['epoch'], 'configuration': vars(args), 'seed': 42, 'device': str(device), 'data_audit': audit, 'validation_trials': history, 'threshold_trials': threshold_trials, 'uncertainty_threshold': threshold, 'training_seconds': time.perf_counter() - started, 'test': {'accuracy': accuracy_score(labels, predicted), 'macro_f1': f1_score(labels, predicted, average='macro'), 'classification_report': classification_report(labels, predicted, labels=list(range(6)), target_names=LABELS, output_dict=True, zero_division=0), 'labels': LABELS, 'confusion_matrix': confusion_matrix(labels, predicted, labels=list(range(6))).tolist()}, 'limitations': ['Partial fine-tuning: lower encoder layers frozen to fit CPU resources.', 'One random seed and short-text benchmark; no independent journal validation.', 'Uncalibrated probabilities; no neutral class.']}
    (ROOT / 'artifacts' / 'distilbert_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'metadata.json').write_text(json.dumps({'version': version, 'threshold': threshold, 'weights_sha256': hashlib.sha256((output / 'model.safetensors').read_bytes()).hexdigest()}, indent=2), encoding='utf-8')
    print(json.dumps(report['test'], indent=2), flush=True)


if __name__ == '__main__':
    main()
