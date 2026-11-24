"""Package trusted trained artifacts for GitHub Releases without journal data."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / 'transfer' / 'MoodJournal-models-v1.zip'
    output.parent.mkdir(exist_ok=True)
    candidates = [ROOT / 'artifacts/baseline.joblib']
    for folder in ['distilbert', 'distilbert_continued', 'distilbert_experiment_2']:
        candidates.extend(p for p in (ROOT / 'artifacts' / folder).rglob('*') if p.is_file() and p.name != 'best_weights.pt')
    candidates.extend((ROOT / 'artifacts').glob('*.json'))
    candidates.extend((ROOT / 'artifacts/pilot_evaluation').glob('*.json'))
    manifest = {'scope': 'Trusted local models, tokenizer files, reports and continuation checkpoint. No journals, credentials, caches or original dataset archives.', 'files': {}}
    with ZipFile(output, 'w', compression=ZIP_DEFLATED, compresslevel=3) as z:
        for path in sorted(set(candidates)):
            if path.suffix not in {'.json', '.joblib', '.safetensors', '.pt'}:
                continue
            relative = path.relative_to(ROOT).as_posix()
            manifest['files'][relative] = {'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(), 'bytes': path.stat().st_size}
            z.write(path, relative)
        z.writestr('model_bundle_manifest.json', json.dumps(manifest, indent=2))
    with ZipFile(output) as z:
        assert z.testzip() is None
    print(json.dumps({'path': str(output), 'bytes': output.stat().st_size, 'files': len(manifest['files']), 'sha256': hashlib.file_digest(output.open('rb'), 'sha256').hexdigest()}), flush=True)


if __name__ == '__main__':
    main()
