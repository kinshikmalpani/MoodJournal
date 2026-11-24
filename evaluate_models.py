"""Diagnostic comparison only. Never reads the personal journal database."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from journal_core import analyse
from model_runtime import load_artifact, LABELS
from model_catalog import available_models, read_report, validation_score

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / 'research' / 'journal_development.jsonl'
    cases = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    models = available_models()
    comparison = {'scope': 'AI-authored development diagnostics; not an independent test or clinical validation.', 'fixture_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'case_count': len(cases), 'models': {}}
    for name in models:
        artifact = load_artifact(name)
        artifact['pipeline'].predict_proba(['Warmup inference.'])
        rows = []
        for case in cases:
            start = time.perf_counter()
            result = analyse(case['text'], artifact)
            elapsed = time.perf_counter() - start
            whole = artifact['pipeline'].predict_proba([case['text']])[0]
            whole_label = str(artifact['pipeline'].classes_[int(np.argmax(whole))])
            rows.append({**case, 'prediction': result['label'], 'top_label': max(result['scores'], key=result['scores'].get), 'whole_entry_top_label': whole_label, 'passage_labels': [p['label'] for p in result['passages']], 'seconds': elapsed, 'scores': result['scores']})
        single = [r for r in rows if len(r['expected_emotions']) == 1]
        neutral = [r for r in rows if r['category'] == 'neutral']
        mixed = [r for r in rows if r['category'] == 'mixed']
        expected = [r['expected_emotions'][0] for r in single]
        report = read_report(name)
        validation = validation_score(report)
        comparison['models'][name] = {'version': artifact['version'], 'benchmark_validation_macro_f1': validation, 'benchmark_test_accuracy': report['test']['accuracy'], 'benchmark_test_macro_f1': report['test']['macro_f1'], 'development_single_label_count': len(single), 'development_chunked_macro_f1': f1_score(expected, [r['top_label'] for r in single], labels=LABELS, average='macro', zero_division=0), 'development_whole_macro_f1': f1_score(expected, [r['whole_entry_top_label'] for r in single], labels=LABELS, average='macro', zero_division=0), 'neutral_abstained': sum(r['prediction'] == 'uncertain' for r in neutral), 'neutral_count': len(neutral), 'mixed_both_detected': sum(set(r['expected_emotions']).issubset(set(r['passage_labels'])) for r in mixed), 'mixed_count': len(mixed), 'median_journal_latency_ms': float(np.median([r['seconds'] for r in rows]) * 1000), 'cases': rows}
    # Deployment choice is based on validation, never short-text test or these fixtures.
    comparison['recommended_model'] = max(models, key=lambda name: comparison['models'][name]['benchmark_validation_macro_f1'])
    comparison['selection_rule'] = 'Highest short-text validation macro-F1 among completed candidates. Deployment recommendation is provisional until independent journal evaluation.'
    (ROOT / 'artifacts' / 'model_comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    print(json.dumps({name: {k: v for k, v in result.items() if k != 'cases'} for name, result in comparison['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
