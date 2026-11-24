"""Frozen external and synthetic evaluations; no training or journal database access."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from journal_core import analyse
from model_runtime import LABELS, load_artifact

DATA = ROOT / 'research/pilot'
OUT = ROOT / 'artifacts/pilot_evaluation'
COMMON = ['anger', 'fear', 'joy', 'sadness']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(target, predicted):
    return {'count': len(target), 'accuracy': accuracy_score(target, predicted), 'macro_f1_supported_four': f1_score(target, predicted, labels=COMMON, average='macro', zero_division=0), 'classification_report': classification_report(target, predicted, labels=LABELS, output_dict=True, zero_division=0)}


def trend_metrics(rows):
    days = defaultdict(list)
    for r in rows:
        intended = r['intended_passages']
        predicted = r['passage_labels']
        truth = np.array([sum(p['label'] == label for p in intended) / len(intended) for label in LABELS])
        estimate = np.array([predicted.count(label) / len(predicted) for label in LABELS])
        days[(r['person'], r['day_index'])].append((truth, estimate))
    daily = {key: (np.mean([p[0] for p in pairs], axis=0), np.mean([p[1] for p in pairs], axis=0)) for key, pairs in days.items()}
    errors, changed, unchanged, daily_rows = [], [], [], []
    direction = lambda x: np.where(x > .05, 1, np.where(x < -.05, -1, 0))
    for (person, day), (truth, estimate) in sorted(daily.items()):
        errors.extend(abs(truth - estimate))
        daily_rows.append({'person': person, 'day_index': day, 'intended_frequency': dict(zip(LABELS, truth.tolist())), 'predicted_frequency': dict(zip(LABELS, estimate.tolist())), 'entry_count': len(days[(person, day)])})
        prior = daily.get((person, day - 1))
        if prior is not None:
            actual = direction(truth - prior[0])
            inferred = direction(estimate - prior[1])
            changed.extend((actual == inferred)[actual != 0].tolist())
            unchanged.extend((actual == inferred)[actual == 0].tolist())
    return {'observed_person_days': len(days), 'daily_frequency_mae': float(np.mean(errors)), 'changed_class_transitions': len(changed), 'changed_direction_accuracy': float(np.mean(changed)) if changed else None, 'unchanged_class_transitions': len(unchanged), 'unchanged_direction_accuracy': float(np.mean(unchanged)) if unchanged else None, 'daily': daily_rows}


def main():
    protocol = json.loads((DATA / 'protocol.json').read_text())
    for name, expected in protocol['files'].items():
        assert sha(DATA / name) == expected, f'Data changed: {name}'
    for name, info in protocol['models'].items():
        assert sha(ROOT / info['path']) == info['sha256'], f'Model changed: {name}'
    OUT.mkdir(exist_ok=True)
    freeze = OUT / 'evaluation_freeze.json'
    tracked = [ROOT / 'research/evaluate_pilot.py', ROOT / 'model_runtime.py', ROOT / 'journal_core.py', DATA / 'protocol.json']
    for model in protocol['models']:
        if model != 'baseline':
            folder = (ROOT / protocol['models'][model]['path']).parent
            tracked.extend(p for p in folder.iterdir() if p.is_file() and p.suffix == '.json')
            tracked.append(ROOT / 'artifacts' / model / 'metadata.json')
    current = {str(p.relative_to(ROOT)): sha(p) for p in tracked}
    if freeze.exists():
        assert json.loads(freeze.read_text()) == current, 'Evaluation code or configuration changed after freeze.'
    else:
        freeze.write_text(json.dumps(current, indent=2))
    external = [json.loads(line) for line in (DATA / 'external.jsonl').read_text(encoding='utf-8').splitlines()]
    synthetic = [json.loads(line) for line in (DATA / 'synthetic.jsonl').read_text(encoding='utf-8').splitlines()]
    reports = {}
    for name in protocol['models']:
        print(f'Evaluating {name}', flush=True)
        artifact = load_artifact(name)
        predictor = artifact['pipeline']
        whole = predictor.predict_proba([r['text'] for r in external])
        records = []
        for i, r in enumerate(external):
            result = analyse(r['text'], artifact)
            records.append({'id': r['id'], 'author_group': r['author_group'], 'target': r['label'], 'supported': r['supported'], 'whole': str(predictor.classes_[whole[i].argmax()]), 'passage_mean': max(result['scores'], key=result['scores'].get), 'summary': result['label']})
            if (i + 1) % 200 == 0:
                print(f'{name}: {i + 1}/{len(external)} external entries', flush=True)
        supported = [r for r in records if r['supported']]
        unsupported = [r for r in records if not r['supported']]
        targets = [r['target'] for r in supported]
        synthetic_rows = []
        for r in synthetic:
            result = analyse(r['text'], artifact)
            synthetic_rows.append({**r, 'prediction': result['label'], 'passage_labels': [p['label'] for p in result['passages']], 'detected_emotions': result['detected_emotions'], 'scores': result['scores']})
        neutral = [r for r in synthetic_rows if not r['expected_emotions']]
        mixed = [r for r in synthetic_rows if len(r['expected_emotions']) > 1]
        report = {'version': artifact['version'], 'external': {'passage_mean': metrics(targets, [r['passage_mean'] for r in supported]), 'whole': metrics(targets, [r['whole'] for r in supported]), 'unsupported_count': len(unsupported), 'unsupported_abstained': sum(r['summary'] == 'uncertain' for r in unsupported), 'predictions': records}, 'synthetic': {'count': len(synthetic_rows), 'neutral_count': len(neutral), 'neutral_abstained': sum(r['prediction'] == 'uncertain' for r in neutral), 'mixed_count': len(mixed), 'mixed_exact_set_correct': sum(set(r['detected_emotions']) == set(r['expected_emotions']) for r in mixed), 'trend': trend_metrics(synthetic_rows), 'predictions': synthetic_rows}}
        (OUT / f'{name}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        reports[name] = { 'external_passage_macro_f1': report['external']['passage_mean']['macro_f1_supported_four'], 'external_whole_macro_f1': report['external']['whole']['macro_f1_supported_four'], 'external_count': len(supported), 'synthetic_neutral_abstained': report['synthetic']['neutral_abstained'], 'synthetic_neutral_count': len(neutral), 'synthetic_mixed_exact': report['synthetic']['mixed_exact_set_correct'], 'synthetic_mixed_count': len(mixed), **{k: v for k, v in report['synthetic']['trend'].items() if k != 'daily'}}
        print(json.dumps(reports[name]), flush=True)
        del artifact, predictor
    # Paired bootstrap by source author; uncertainty conditional on this corpus.
    predictions = {name: json.loads((OUT / f'{name}.json').read_text())['external']['predictions'] for name in reports}
    base = [r for r in predictions['baseline'] if r['supported']]
    v2 = [r for r in predictions['distilbert_experiment_2'] if r['supported']]
    assert [r['id'] for r in base] == [r['id'] for r in v2]
    groups = sorted({r['author_group'] for r in base})
    indices = {g: [i for i, r in enumerate(base) if r['author_group'] == g] for g in groups}
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(1000):
        idx = [i for g in rng.choice(groups, len(groups), replace=True) for i in indices[g]]
        y = [base[i]['target'] for i in idx]
        deltas.append(f1_score(y, [v2[i]['passage_mean'] for i in idx], labels=COMMON, average='macro', zero_division=0) - f1_score(y, [base[i]['passage_mean'] for i in idx], labels=COMMON, average='macro', zero_division=0))
    summary = {'scope': 'External event-text evaluation and AI-authored synthetic longitudinal stress test; not a human journal pilot.', 'models': reports, 'v2_minus_baseline_external_passage_macro_f1_bootstrap_95_percent_interval': np.quantile(deltas, [.025, .975]).tolist(), 'bootstrap_author_groups': len(groups), 'bootstrap_replicates': 1000, 'protocol_sha256': sha(DATA / 'protocol.json')}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
