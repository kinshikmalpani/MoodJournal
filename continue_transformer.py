"""New warm-start experiment; subsequent interruptions resume full state exactly on CPU."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from filelock import FileLock
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from train_transformer import ROOT, EncodedDataset, LengthGroupedBatches
from train_baseline import prepare_data
from model_runtime import LABELS
from training_state import atomic_json, save_checkpoint, load_checkpoint

OUTPUT = ROOT / 'artifacts' / 'distilbert_continued'


def run(args):
    source = ROOT / 'artifacts' / 'distilbert'
    checkpoint_file = OUTPUT / 'training_checkpoint.pt'
    if (OUTPUT / 'metadata.json').exists():
        print('Experiment already complete; existing results preserved.', flush=True)
        return
    if checkpoint_file.exists() != args.resume:
        raise ValueError('Use --resume for an existing checkpoint; omit it only for a new experiment.')
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(source, local_files_only=True)
    for p in model.distilbert.parameters():
        p.requires_grad = False
    for layer in model.distilbert.transformer.layer[-2:]:
        for p in layer.parameters():
            p.requires_grad = True
    splits, audit = prepare_data()
    train = EncodedDataset(splits['train'], tokenizer)
    validation = EncodedDataset(splits['val'], tokenizer)
    source_hash = hashlib.sha256((source / 'model.safetensors').read_bytes()).hexdigest()
    configuration = {'max_additional_epochs': 3, 'patience': 1, 'min_delta': .002, 'learning_rate': 2e-5, 'batch_size': 32, 'trainable_layers': 2, 'max_tokens': 96, 'seed': 42, 'checkpoint_every_steps': 50, 'source_weights_sha256': source_hash, 'dataset_sha256': audit['archive_sha256']}
    steps_per_epoch = (len(train) + 31) // 32
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=configuration['learning_rate'], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=0, threshold=.002, threshold_mode='abs')
    baseline_report = json.loads((ROOT / 'artifacts' / 'distilbert_report.json').read_text(encoding='utf-8'))
    initial_score = max(r['validation_macro_f1'] for r in baseline_report['validation_trials'])
    state = {'configuration': configuration, 'epoch': 0, 'next_batch': 0, 'batch_order': None, 'global_step': 0, 'epoch_loss': 0., 'best_score': initial_score, 'best_epoch': 0, 'patience_score': initial_score, 'bad_epochs': 0, 'history': [{'additional_epoch': 0, 'validation_macro_f1': initial_score, 'note': 'Starting saved first-epoch model.'}], 'elapsed_seconds': 0., 'phase': 'training'}
    if args.resume:
        state = load_checkpoint(checkpoint_file, model, optimizer, scheduler)
        if state['configuration'] != configuration:
            raise ValueError('Source, dataset, or training configuration changed since checkpoint.')
        print(f'Resumed global step {state["global_step"]}, next batch {state["next_batch"]}', flush=True)
    else:
        # Initial weights also participate in validation-only selection.
        model.save_pretrained(OUTPUT / 'best', safe_serialization=True)
        tokenizer.save_pretrained(OUTPUT / 'best')
        save_checkpoint(checkpoint_file, model, optimizer, scheduler, state)
        atomic_json(OUTPUT / 'run_configuration.json', configuration)
    started, previous_elapsed = time.perf_counter(), state['elapsed_seconds']

    def save():
        state['elapsed_seconds'] = previous_elapsed + time.perf_counter() - started
        save_checkpoint(checkpoint_file, model, optimizer, scheduler, state)

    def progress(phase):
        atomic_json(OUTPUT / 'progress.json', {'phase': phase, 'additional_epoch': state['epoch'] + 1, 'next_batch': state['next_batch'], 'steps_per_epoch': steps_per_epoch, 'global_step': state['global_step'], 'best_validation_macro_f1': state['best_score'], 'elapsed_seconds': round(previous_elapsed + time.perf_counter() - started), 'max_additional_epochs': 3})

    def collate(dataset, indices):
        rows = [dataset[i] for i in indices]
        inputs = tokenizer.pad([{k: v for k, v in row.items() if k != 'labels'} for row in rows], return_tensors='pt')
        return inputs, torch.tensor([row['labels'] for row in rows])

    def predict(dataset):
        model.eval()
        values = np.zeros((len(dataset), 6))
        with torch.inference_mode():
            for indices in LengthGroupedBatches(dataset, 32, False):
                inputs, _ = collate(dataset, indices)
                values[indices] = model(**inputs).logits.softmax(-1).numpy()
        return values

    while state['phase'] == 'training' and state['epoch'] < 3:
        if state['batch_order'] is None:
            state['batch_order'] = list(LengthGroupedBatches(train, 32, True))
            save()
        model.train()
        model.distilbert.embeddings.eval()
        for layer in model.distilbert.transformer.layer[:-2]:
            layer.eval()
        for index in range(state['next_batch'], len(state['batch_order'])):
            inputs, targets = collate(train, state['batch_order'][index])
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(**inputs).logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            state['global_step'] += 1
            state['next_batch'] = index + 1
            state['epoch_loss'] += loss.item()
            if state['global_step'] % 10 == 0:
                progress('training')
                print(f'Epoch +{state["epoch"] + 1}, batch {index + 1}/{steps_per_epoch}, mean loss {state["epoch_loss"] / (index + 1):.4f}', flush=True)
            if state['global_step'] % 50 == 0:
                save()
            if args.stop_after_steps and state['global_step'] >= args.stop_after_steps:
                save()
                progress('paused_for_resume_check')
                print('Intentional stop after saving resumable state.', flush=True)
                return
        # Save before validation so an interruption cannot lose this epoch's updates.
        save()
        progress('validation')
        values = predict(validation)
        score = f1_score(validation.labels, values.argmax(1), average='macro')
        epoch = state['epoch'] + 1
        state['history'].append({'additional_epoch': epoch, 'validation_macro_f1': score, 'training_loss': state['epoch_loss'] / steps_per_epoch})
        if score > state['best_score']:
            # Use an atomic weight file for best selection; checkpoint stores its identity.
            best_temp = OUTPUT / 'best_weights.tmp'
            torch.save(model.state_dict(), best_temp)
            best_temp.replace(OUTPUT / 'best_weights.pt')
            state['best_score'], state['best_epoch'] = score, epoch
        if score > state['patience_score'] + configuration['min_delta']:
            state['patience_score'], state['bad_epochs'] = score, 0
        else:
            state['bad_epochs'] += 1
        scheduler.step(score)
        state['epoch'] += 1
        state['batch_order'], state['next_batch'], state['epoch_loss'] = None, 0, 0.
        if state['bad_epochs'] >= configuration['patience'] or state['epoch'] >= 3:
            state['phase'] = 'final_evaluation'
        atomic_json(OUTPUT / 'history.json', state['history'])
        save()
        print(f'Validation epoch +{epoch}: {score:.4f}; best {state["best_score"]:.4f}', flush=True)

    progress('final_evaluation')
    if state['best_epoch'] > 0:
        model.load_state_dict(torch.load(OUTPUT / 'best_weights.pt', map_location='cpu', weights_only=True))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(source, local_files_only=True)
    val = predict(validation)
    trials = []
    for threshold in np.arange(.35, .86, .05):
        keep = val.max(1) >= threshold
        trials.append({'threshold': round(float(threshold), 2), 'coverage': float(keep.mean()), 'accuracy': float((val.argmax(1)[keep] == np.array(validation.labels)[keep]).mean()) if keep.any() else 0.})
    eligible = [row for row in trials if row['accuracy'] >= .85 and row['coverage'] >= .3]
    threshold = max(eligible, key=lambda row: row['coverage'])['threshold'] if eligible else .5
    test_dataset = EncodedDataset(splits['test'], tokenizer)
    test_probs = predict(test_dataset)
    pred, targets = test_probs.argmax(1), test_dataset.labels
    model.save_pretrained(OUTPUT / 'best', safe_serialization=True)
    report = {'model_version': f'distilbert-continued-v1-epoch{state["best_epoch"]}', 'training_status': 'New continuation experiment from saved first-epoch weights. Original optimizer/scheduler were unavailable and reset; subsequent resumes restore full state.', 'configuration': configuration, 'selected_additional_epoch': state['best_epoch'], 'completed_additional_epochs': state['epoch'], 'stop_reason': 'validation_patience' if state['bad_epochs'] else 'maximum_epochs', 'data_audit': audit, 'validation_trials': state['history'], 'threshold_trials': trials, 'uncertainty_threshold': threshold, 'test': {'accuracy': accuracy_score(targets, pred), 'macro_f1': f1_score(targets, pred, average='macro'), 'classification_report': classification_report(targets, pred, labels=list(range(6)), target_names=LABELS, output_dict=True, zero_division=0), 'labels': LABELS, 'confusion_matrix': confusion_matrix(targets, pred, labels=list(range(6))).tolist()}, 'limitations': ['Continuation with a fresh optimizer, not exact resume of original interrupted run.', 'Single seed, frozen lower encoder layers, no independent journal validation.', 'Original test set previously reported; this is a repeated benchmark comparison, not a new untouched external test.']}
    atomic_json(ROOT / 'artifacts' / 'distilbert_continued_report.json', report)
    atomic_json(OUTPUT / 'metadata.json', {'version': report['model_version'], 'threshold': threshold, 'weights_sha256': hashlib.sha256((OUTPUT / 'best' / 'model.safetensors').read_bytes()).hexdigest()})
    state['phase'] = 'complete'
    progress('complete')
    print(json.dumps({'test_accuracy': report['test']['accuracy'], 'test_macro_f1': report['test']['macro_f1'], 'best_validation_macro_f1': state['best_score']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-steps', type=int, default=0, help='Intentional checkpoint/resume verification; normally omit.')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUTPUT / '.training.lock'), timeout=0):
        run(args)


if __name__ == '__main__':
    main()
