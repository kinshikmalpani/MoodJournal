"""Evaluate a saved checkpoint without restarting training or reading journals."""
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from train_transformer import ROOT, EncodedDataset, LengthGroupedBatches
from train_baseline import prepare_data
from model_runtime import LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recovered-epoch', type=int, choices=[1], required=True)
    args = parser.parse_args()
    folder = ROOT / 'artifacts' / 'distilbert'
    if (folder / 'metadata.json').exists():
        raise RuntimeError('This checkpoint has already been finalized.')
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(folder, local_files_only=True)
    model.eval()
    splits, audit = prepare_data()
    started = time.perf_counter()
    def predict(name):
        dataset = EncodedDataset(splits[name], tokenizer)
        sampler = LengthGroupedBatches(dataset, 32, False)
        def collate(rows):
            return tokenizer.pad([{k: v for k, v in row.items() if k != 'labels'} for row in rows], return_tensors='pt')
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate)
        probs = np.zeros((len(dataset), len(LABELS)))
        with torch.inference_mode():
            for step, (indices, inputs) in enumerate(zip(sampler, loader)):
                probs[indices] = model(**inputs).logits.softmax(-1).numpy()
                if step % 20 == 0:
                    print(f'{name}: batch {step + 1}/{len(sampler)}', flush=True)
        return np.array(dataset.labels), probs
    y_val, val = predict('val')
    val_f1 = f1_score(y_val, val.argmax(1), average='macro')
    trials = []
    for value in np.arange(.35, .86, .05):
        keep = val.max(1) >= value
        trials.append({'threshold': round(float(value), 2), 'coverage': float(keep.mean()), 'accuracy': float((val.argmax(1)[keep] == y_val[keep]).mean()) if keep.any() else 0.})
    eligible = [r for r in trials if r['accuracy'] >= .85 and r['coverage'] >= .3]
    threshold = max(eligible, key=lambda r: r['coverage'])['threshold'] if eligible else .5
    print(f'Validation macro-F1: {val_f1:.4f}; threshold: {threshold}', flush=True)
    y_test, test = predict('test')
    pred = test.argmax(1)
    report = {
        'model_version': 'distilbert-partial-epoch1-v1',
        'checkpoint': 'distilbert/distilbert-base-uncased',
        'selected_epoch': args.recovered_epoch,
        'training_status': 'Recovered completed first-epoch checkpoint after interruption during epoch two. Later unsaved updates were lost. No completed two-epoch comparison is claimed.',
        'configuration': {'planned_epochs': 2, 'saved_checkpoint_epoch': args.recovered_epoch, 'batch_size': 32, 'threads': 4, 'trainable_layers': 2, 'training_max_tokens': 96, 'learning_rate': 3e-5, 'batching': 'length-grouped, shuffled batch order'},
        'seed': 42, 'device': 'cpu', 'data_audit': audit,
        'validation_trials': [{'epoch': args.recovered_epoch, 'validation_macro_f1': val_f1, 'note': 'Re-evaluated saved checkpoint; original training log was not persisted.'}],
        'threshold_trials': trials, 'uncertainty_threshold': threshold,
        'evaluation_seconds': time.perf_counter() - started,
        'test': {'accuracy': accuracy_score(y_test, pred), 'macro_f1': f1_score(y_test, pred, average='macro'), 'classification_report': classification_report(y_test, pred, labels=list(range(6)), target_names=LABELS, output_dict=True, zero_division=0), 'labels': LABELS, 'confusion_matrix': confusion_matrix(y_test, pred, labels=list(range(6))).tolist()},
        'limitations': ['One completed epoch of partial fine-tuning; not a completed two-epoch experiment.', 'One seed; short-text benchmark only, no independently labelled journal test.', 'Uncalibrated probabilities; no neutral class.']}
    # Persist report first; metadata is the ready-to-use marker for the app.
    (ROOT / 'artifacts' / 'distilbert_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (folder / 'metadata.json').write_text(json.dumps({'version': report['model_version'], 'threshold': threshold, 'weights_sha256': hashlib.sha256((folder / 'model.safetensors').read_bytes()).hexdigest()}, indent=2), encoding='utf-8')
    print(json.dumps({'validation_macro_f1': val_f1, 'test_accuracy': report['test']['accuracy'], 'test_macro_f1': report['test']['macro_f1']}), flush=True)


if __name__ == '__main__':
    main()
