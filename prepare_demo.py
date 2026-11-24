"""Seed an isolated presentation journal with clearly fictional examples."""
import datetime as dt
from journal_core import ROOT, today, load_entries, save_entry, analyse
from model_catalog import recommended_model
from model_runtime import load_artifact


def main():
    database = ROOT / 'local_data' / 'demo.sqlite3'
    if load_entries(path=database):
        print('Existing demo entries preserved.')
        return
    artifact = load_artifact(recommended_model())
    examples = [
        'I am excited and happy that our project is finally working.',
        'I feel nervous about presenting in front of the class.',
        'I miss my old friends and feel sad today.',
        'I am furious that someone took credit for my work.',
        'I adore my family and feel loved when they support me.',
        'I was astonished by the unexpected birthday celebration.',
        'I am worried about the examination. I am delighted that my project works.',
    ]
    for offset, text in enumerate(examples):
        save_entry(text, analyse(text, artifact), day=today()-dt.timedelta(days=6-offset), path=database)
    print(f'Created {len(examples)} fictional entries in {database}. Personal journal untouched.')


if __name__ == '__main__':
    main()
