"""One registry for inference, evaluation, and the presentation UI."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = {
    'baseline': ('TF-IDF baseline', 'baseline_report.json', None),
    'distilbert': ('DistilBERT — first epoch', 'distilbert_report.json', 'distilbert'),
    'distilbert_continued': ('DistilBERT — continued', 'distilbert_continued_report.json', 'distilbert_continued/best'),
    'distilbert_experiment_2': ('DistilBERT v2 — final prototype', 'distilbert_experiment_2/report.json', 'distilbert_experiment_2/best_model'),
}


def read_report(name):
    return json.loads((ROOT / 'artifacts' / MODELS[name][1]).read_text(encoding='utf-8'))


def validation_score(report):
    if 'best_validation_macro_f1' in report:
        return report['best_validation_macro_f1']
    return max(row['validation_macro_f1'] for row in report['validation_trials'])


def available_models():
    available = []
    for name, (_, report, weights) in MODELS.items():
        artifact = ROOT / 'artifacts'
        if not (artifact / report).is_file():
            continue
        if weights is None:
            ready = (artifact / 'baseline.joblib').is_file()
        else:
            ready = (artifact / name / 'metadata.json').is_file() and (artifact / weights / 'model.safetensors').is_file()
        if ready:
            available.append(name)
    return available


def recommended_model(names=None):
    names = available_models() if names is None else names
    return max(names, key=lambda name: validation_score(read_report(name)))
