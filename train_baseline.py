"""Train only on the original archive; select settings on validation, test once."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from collections import Counter, defaultdict
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent


def prepare_data():
    archive = ROOT / 'Emotions Dataset.zip'
    splits, audit, seen = {}, {}, set()
    # Give test precedence, then validation: remove overlaps from earlier learning stages.
    with ZipFile(archive) as z:
        for split in ('test', 'val', 'train'):
            rows = [line.rsplit(';', 1) for line in z.read(f'Emotions Dataset/{split}.txt').decode('utf-8-sig').splitlines() if line.strip()]
            grouped = defaultdict(list)
            for text, label in rows:
                grouped[' '.join(text.lower().split())].append((text, label))
            clean = []
            conflicts = overlaps = 0
            for key, values in grouped.items():
                if key in seen:
                    overlaps += len(values)
                elif len({label for _, label in values}) > 1:
                    conflicts += len(values)
                else:
                    clean.append(values[0])
            seen.update(grouped)
            splits[split] = clean
            audit[split] = {'original_rows': len(rows), 'retained_rows': len(clean), 'overlap_rows_removed': overlaps, 'conflicting_rows_removed': conflicts, 'class_counts': dict(Counter(y for _, y in clean))}
    audit['archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    audit['policy'] = 'Normalize whitespace/case; preserve test before validation before train; deduplicate within split and exclude conflicting groups. No data.zip used.'
    return splits, audit


def main():
    splits, audit = prepare_data()
    x_train, y_train = zip(*splits['train'])
    x_val, y_val = zip(*splits['val'])
    trials, best, best_score = [], None, -1
    for weight in (None, 'balanced'):
        model = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=60000)), ('classifier', LogisticRegression(C=4, class_weight=weight, max_iter=1000, random_state=42))])
        model.fit(x_train, y_train)
        score = f1_score(y_val, model.predict(x_val), average='macro')
        trials.append({'class_weight': weight, 'validation_macro_f1': score})
        print(trials[-1], flush=True)
        if score > best_score:
            best, best_score = model, score
    # Choose an abstention threshold using validation only. Not a neutral detector.
    probs = best.predict_proba(x_val)
    predicted = best.classes_[probs.argmax(axis=1)]
    threshold_trials = []
    for threshold in np.arange(.35, .86, .05):
        keep = probs.max(axis=1) >= threshold
        threshold_trials.append({'threshold': round(float(threshold), 2), 'coverage': float(keep.mean()), 'accuracy': float((predicted[keep] == np.array(y_val)[keep]).mean()) if keep.any() else 0.0})
    eligible = [r for r in threshold_trials if r['accuracy'] >= .85 and r['coverage'] >= .3]
    threshold = max(eligible, key=lambda r: r['coverage'])['threshold'] if eligible else .5
    x_test, y_test = zip(*splits['test'])
    pred = best.predict(x_test)
    report = {'model_version': 'tfidf-logreg-v1', 'data_audit': audit, 'validation_trials': trials, 'threshold_trials': threshold_trials, 'uncertainty_threshold': threshold, 'test': {'accuracy': accuracy_score(y_test, pred), 'macro_f1': f1_score(y_test, pred, average='macro'), 'classification_report': classification_report(y_test, pred, output_dict=True), 'labels': best.classes_.tolist(), 'confusion_matrix': confusion_matrix(y_test, pred, labels=best.classes_).tolist()}, 'limitations': ['Original short-text test only; not independent journal validation.', 'Softmax/class probabilities are uncalibrated model scores, not emotion intensity.', 'Threshold validated on whole short texts, not journal chunks.', 'No neutral class or clinical interpretation.']}
    output = ROOT / 'artifacts'
    output.mkdir(exist_ok=True)
    joblib.dump({'pipeline': best, 'threshold': threshold, 'version': report['model_version']}, output / 'baseline.joblib')
    (output / 'baseline_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['test'], indent=2))


if __name__ == '__main__':
    main()
