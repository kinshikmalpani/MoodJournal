"""Make a portable archive without Windows dependencies or model download caches."""
import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    transfer = ROOT / 'transfer'
    transfer.mkdir(exist_ok=True)
    output = transfer / 'MoodJournal-Mac.zip'
    pending = transfer / 'MoodJournal-Mac.zip.partial'
    files = [p for p in ROOT.iterdir() if p.is_file() and p.suffix.lower() in {'.py', '.md', '.txt', '.ps1', '.zip'}]
    for directory in ['.streamlit', 'research', 'tests', 'artifacts']:
        files.extend(p for p in (ROOT / directory).rglob('*') if p.is_file() and not any(part in {'hf_cache', '__pycache__'} for part in p.relative_to(ROOT).parts) and p.suffix not in {'.tmp', '.lock', '.pyc'})
    manifest = {'description': 'Portable MoodJournal project. Includes private journal snapshot; no Windows virtualenv or model download cache.', 'files': {}}
    journal_snapshot = transfer / 'journal-transfer.sqlite3'
    journal = ROOT / 'local_data' / 'journal.sqlite3'
    if journal.exists():
        with sqlite3.connect(f'file:{journal.as_posix()}?mode=ro', uri=True) as source, sqlite3.connect(journal_snapshot) as destination:
            source.backup(destination)
        source.close()
        destination.close()
    try:
        with zipfile.ZipFile(pending, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            entries = [(p, p.relative_to(ROOT).as_posix()) for p in sorted(set(files))]
            if journal_snapshot.exists():
                entries.append((journal_snapshot, 'local_data/journal.sqlite3'))
            for path, relative in entries:
                digest = hashlib.sha256()
                with path.open('rb') as source, archive.open(f'MoodJournal/{relative}', 'w', force_zip64=True) as target:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        target.write(block)
                        digest.update(block)
                manifest['files'][relative] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
                if path.stat().st_size > 100_000_000:
                    print(f'Packaged {relative}', flush=True)
            archive.writestr('MoodJournal/TRANSFER_MANIFEST.json', json.dumps(manifest, indent=2))
        with zipfile.ZipFile(pending) as archive:
            corrupt = archive.testzip()
            if corrupt:
                raise RuntimeError(f'Archive integrity failure: {corrupt}')
        pending.replace(output)
        print(json.dumps({'archive': str(output), 'size_mb': round(output.stat().st_size / 1024 ** 2, 1), 'files': len(manifest['files']), 'integrity': 'passed'}), flush=True)
    finally:
        if journal_snapshot.exists():
            journal_snapshot.unlink()


if __name__ == '__main__':
    main()
