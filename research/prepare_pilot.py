"""Prepare evaluation-only data; never access personal journals or model predictions."""
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_catalog import MODELS

OUT = ROOT / 'research' / 'pilot'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    if OUT.exists():
        raise RuntimeError('Frozen pilot folder exists; use a new version rather than overwrite.')
    archive = ROOT / 'research/external/deISEARenISEAR.zip'
    with ZipFile(archive) as z:
        source = list(csv.DictReader(io.StringIO(z.read('deISEARenISEAR/enISEAR.tsv').decode()), delimiter='\t'))
        licence = z.read('deISEARenISEAR/license.txt')
        readme = z.read('deISEARenISEAR/readme.txt')
    normalize = lambda s: ' '.join(s.lower().split())
    with ZipFile(ROOT / 'Emotions Dataset.zip') as z:
        original = {normalize(line.rsplit(';', 1)[0]) for split in ['train', 'val', 'test'] for line in z.read(f'Emotions Dataset/{split}.txt').decode('utf-8-sig').splitlines() if line.strip()}
    seen, external, excluded = set(), [], []
    for r in source:
        text = r['Sentence'].strip()
        key = normalize(text)
        reason = 'source_benchmark_overlap' if key in original else 'external_duplicate' if key in seen else None
        seen.add(key)
        if reason:
            excluded.append({'id': r['Sentence_id'], 'reason': reason})
            continue
        label = r['Prior_Emotion'].lower()
        external.append({'id': 'enisear-' + r['Sentence_id'], 'author_group': r['Worker_id'], 'text': text, 'label': label, 'supported': label in ['anger', 'fear', 'joy', 'sadness'], 'reader_votes': {k.lower(): int(r[k]) for k in ['Anger', 'Disgust', 'Fear', 'Guilt', 'Joy', 'Sadness', 'Shame']}, 'provenance': 'Human-written prompted event description; original author prompt label; not a dated journal.'})

    # Each tuple is one fictional day: passages have author-intended labels.
    # No model outputs were consulted to write these cases. Labels are NOT human annotations.
    stories = {
        'student': [
            [('fear', 'The presentation is a week away and my stomach tightens whenever I imagine standing at the front.'), ('fear', 'I keep rehearsing what might go wrong instead of sleeping.')],
            [('fear', 'I opened the slides again after dinner, convinced that I would forget every line.'), ('joy', 'Finishing the first practice run gave me a small reason to celebrate.')],
            [('joy', 'My teammate listened to the rehearsal and I walked home grinning.'), ('fear', 'There is still a knot in my chest when I picture the questions.')],
            [('neutral', 'I printed twelve pages and put them in the blue folder.'), ('neutral', 'The practice room is booked for Thursday afternoon.')],
            [('joy', 'I am not frightened of presenting now; I am genuinely excited to show our work.'), ('joy', 'We finally made the demonstration work and I wanted to cheer.')],
            [('joy', 'The audience followed our demonstration and I could not stop smiling afterwards.'), ('love', 'My family stayed up to hear how it went, and their care meant so much to me.')],
            [('neutral', 'I returned the adapter to the laboratory and copied the files to a drive.'), ('joy', 'Remembering the applause still makes this ordinary afternoon feel bright.')],
        ],
        'moving': [
            [('sadness', 'The empty chair at our old kitchen table made me cry when I packed the last box.'), ('sadness', 'Leaving my friends behind feels like losing a piece of my life.')],
            [('sadness', 'I ate dinner alone in the new flat and missed the familiar voices down the hallway.'), ('neutral', 'The plates are on the lower shelf beside the sink.')],
            [('love', 'My sister called just to keep me company, and I felt deeply cared for.'), ('sadness', 'After we hung up, the silence brought back how much I miss home.')],
            [('neutral', 'I took the bus to the library and registered for a card.'), ('neutral', 'The return journey took twenty minutes.')],
            [('joy', 'A neighbour invited me for tea and we laughed until the cups were cold.'), ('love', 'Being welcomed so warmly made me feel connected to someone here.')],
            [('joy', 'I found myself humming on the walk back from the market.'), ('joy', 'For the first time this place feels like somewhere I can enjoy living.')],
            [('love', 'I sent my sister a long message about how much her support matters to me.'), ('joy', 'I am looking forward to cooking for my new neighbours this weekend.')],
        ],
        'work': [
            [('neutral', 'I checked the stock list and entered the totals into the spreadsheet.'), ('neutral', 'The meeting starts at ten tomorrow.')],
            [('anger', 'Someone presented my work as their own and I was furious all afternoon.'), ('anger', 'The unfairness kept coming back to me on the train.')],
            [('anger', 'My manager dismissed the problem without listening and I wanted to shout.'), ('sadness', 'By evening I felt defeated, as though my effort did not matter.')],
            [('fear', 'Tomorrow I have to raise the issue again and I am worried about the reaction.'), ('anger', 'I still resent having to fight for credit that should have been mine.')],
            [('joy', 'The team corrected the record and I felt like celebrating.'), ('joy', 'I am not angry with them now; I am delighted that they listened.')],
            [('surprise', 'I opened an ordinary meeting invitation and discovered they had arranged an award for me.'), ('surprise', 'I sat staring at the screen because I had never expected that.')],
            [('neutral', 'I filed the project notes and cleared the desk before leaving.'), ('love', 'My partner brought dinner without being asked and I felt cherished.')],
        ],
        'ordinary': [
            [('neutral', 'I bought rice, soap and a notebook on the way home.'), ('neutral', 'The receipt is in the side pocket of my bag.')],
            [('surprise', 'An old school friend appeared at the door with no warning, leaving me speechless.'), ('joy', 'Once I recognised her, we spent the evening laughing about our school days.')],
            [('love', 'We talked about how we had always looked out for each other, and I treasured that bond.'), ('neutral', 'Her train leaves from platform four in the morning.')],
            [('sadness', 'After she left I sat on the steps feeling the ache of another goodbye.'), ('sadness', 'The room seemed painfully empty without her voice.')],
            [('fear', 'A sudden noise downstairs had me imagining someone breaking in.'), ('joy', 'When I found the cat beside the fallen pan, I laughed and felt happy again.')],
            [('neutral', 'The plumber replaced the washer and wrote the date on the invoice.'), ('neutral', 'I put the spare key back in the drawer.')],
            [('joy', 'I spent the afternoon painting and felt content with what I made.'), ('love', 'Giving the picture to my friend reminded me how much I care about her.')],
        ],
    }
    synthetic = []
    for person, days in stories.items():
        for index, parts in enumerate(days):
            day = index + 1
            if (person == 'moving' and day >= 4) or (person == 'work' and day >= 6):
                day += 1  # Missing calendar days; preserve story order without imputation.
            groups = [[p] for p in parts] if index == 2 else [parts]
            for number, group in enumerate(groups):
                synthetic.append({'id': f'{person}-{day}-{number}', 'person': person, 'day_index': day, 'text': ' '.join(t for _, t in group), 'intended_passages': [{'label': label, 'text': text} for label, text in group], 'expected_emotions': sorted({label for label, _ in group if label != 'neutral'}), 'provenance': 'AI-authored fictional scenario and intended labels; not independent human ground truth.'})
    OUT.mkdir()
    (OUT / 'source_license.txt').write_bytes(licence)
    (OUT / 'source_readme.txt').write_bytes(readme)
    for name, rows in [('external.jsonl', external), ('synthetic.jsonl', synthetic)]:
        (OUT / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    models = {}
    for name in ['baseline', 'distilbert_continued', 'distilbert_experiment_2']:
        path = ROOT / 'artifacts' / ('baseline.joblib' if name == 'baseline' else MODELS[name][2] + '/model.safetensors')
        models[name] = {'path': str(path.relative_to(ROOT)), 'sha256': digest(path)}
    protocol = {'version': 1, 'prepared_date': '2026-11-13', 'source_url': 'https://www.romanklinger.de/data-sets/deISEARenISEAR.zip', 'paper': 'https://aclanthology.org/P19-1391/', 'citation': 'Troiano, Pado and Klinger (2019), Crowdsourcing and Validating Event-focused Emotion Corpora for German and English, ACL.', 'licence': 'ODC-By 1.0; original licence and attribution retained', 'source_sha256': digest(archive), 'external_original_count': len(source), 'retained_count': len(external), 'label_counts': dict(Counter(r['label'] for r in external)), 'exclusions': excluded, 'overlap_policy': 'Exact normalized case/whitespace comparison against all raw original benchmark splits; near-duplicates and pretraining exposure not ruled out.', 'primary_endpoint': 'External supported four-class macro-F1, app passage mean; all six model outputs remain possible. Original author prompt labels are targets. No target fine-tuning.', 'secondary_endpoints': 'Whole-entry inference; unsupported-class abstention; synthetic mixed/neutral performance; daily passage-frequency MAE and adjacent-day direction agreement.', 'trend_definition': 'Mean of entry passage-label frequencies per observed day; neutral/uncertain contributes zero to six emotions. Calendar-adjacent pairs only, delta deadband 0.05. This is a diagnostic frequency measure, not the app softmax chart or mood intensity.', 'synthetic_count': len(synthetic), 'synthetic_people': len(stories), 'synthetic_limit': 'AI-authored non-independent stress test; no human pilot or clinical validation.', 'models': models, 'files': {n: digest(OUT / n) for n in ['external.jsonl', 'synthetic.jsonl']}, 'no_tuning': True}
    write_json(OUT / 'protocol.json', protocol)
    print(json.dumps({k: v for k, v in protocol.items() if k not in ['models', 'files']}, indent=2))


if __name__ == '__main__':
    main()
