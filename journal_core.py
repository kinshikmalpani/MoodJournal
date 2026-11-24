"""Model-independent persistence, calendar summaries, and baseline inference."""
import datetime as dt
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

ROOT = Path(__file__).resolve().parent
DEMO_MODE = os.environ.get('MOODJOURNAL_DEMO') == '1'
DB_PATH = ROOT / 'local_data' / ('demo.sqlite3' if DEMO_MODE else 'journal.sqlite3')
NEGATIVE = {'anger', 'fear', 'sadness'}


def today():
    return dt.datetime.now(ZoneInfo('Asia/Kolkata')).date()


@contextmanager
def connect(path=DB_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.row_factory = sqlite3.Row
        with con:
            con.execute('CREATE TABLE IF NOT EXISTS entries (id INTEGER PRIMARY KEY, day TEXT NOT NULL, text TEXT NOT NULL, result TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)')
            yield con
    finally:
        con.close()


def save_entry(text, result, day=None, path=DB_PATH):
    if not text.strip():
        raise ValueError('An entry cannot be empty.')
    with connect(path) as con:
        cur = con.execute('INSERT INTO entries(day,text,result) VALUES(?,?,?)', ((day or today()).isoformat(), text, json.dumps(result)))
        return cur.lastrowid


def load_entries(days=None, end=None, path=DB_PATH):
    end = end or today()
    with connect(path) as con:
        if days is None:
            rows = con.execute('SELECT * FROM entries ORDER BY day,id').fetchall()
        else:
            start = end - dt.timedelta(days=days - 1)
            rows = con.execute('SELECT * FROM entries WHERE day BETWEEN ? AND ? ORDER BY day,id', (start.isoformat(), end.isoformat())).fetchall()
    return [{**dict(row), 'result': json.loads(row['result'])} for row in rows]


def delete_entry(entry_id, path=DB_PATH):
    with connect(path) as con:
        con.execute('DELETE FROM entries WHERE id=?', (entry_id,))


def daily_summary(entries):
    grouped = {}
    for entry in entries:
        grouped.setdefault(entry['day'], []).append(entry['result'])
    output = []
    for day, results in sorted(grouped.items()):
        labels = results[0]['scores']
        scores = {label: float(np.mean([r['scores'][label] for r in results])) for label in labels}
        negative = sum(scores.get(e, 0) for e in NEGATIVE)
        output.append({'day': day, **scores, 'negative_score': negative, 'entries': len(results)})
    return output


def negative_streak(daily, end=None):
    by_day = {r['day']: r for r in daily}
    cursor, streak = end or today(), 0
    while cursor.isoformat() in by_day:
        if by_day[cursor.isoformat()]['negative_score'] < .6:
            break
        streak += 1
        cursor -= dt.timedelta(days=1)
    return streak


def passages(text, max_words=80):
    output = []
    for sentence in re.split(r'(?<=[.!?])\s+|\n+', text.strip()):
        words = sentence.split()
        output.extend(' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words))
    return output


def analyse(text, artifact):
    if not text.strip() or not any(c.isalpha() for c in text):
        raise ValueError('Write a few words describing your day.')
    if len(text) > 12000:
        raise ValueError('Please keep each entry below 12,000 characters.')
    parts = passages(text)
    model = artifact['pipeline']
    matrix = model.predict_proba(parts)
    scores = matrix.mean(axis=0)
    labels = model.classes_.tolist()
    threshold = artifact['threshold']
    known = model.named_steps['tfidf'].transform(parts).getnnz(axis=1) > 0 if hasattr(model, 'named_steps') else np.ones(len(parts), dtype=bool)
    def label_for(row, has_known=True):
        return labels[int(np.argmax(row))] if has_known and float(max(row)) >= threshold else 'uncertain'
    passage_results = [{'text': part, 'label': label_for(row, bool(k)), 'score': float(max(row))} for part, row, k in zip(parts, matrix, known)]
    detected = sorted({part['label'] for part in passage_results if part['label'] != 'uncertain'})
    summary_label = 'mixed' if len(detected) >= 2 else label_for(scores, bool(known.any()))
    return {'version': artifact['version'], 'label': summary_label, 'detected_emotions': detected, 'scores': dict(zip(labels, map(float, scores))), 'passages': passage_results, 'method': 'Equal-weight passage mean; mixed when confident passage labels differ. Experimental on journals.'}
