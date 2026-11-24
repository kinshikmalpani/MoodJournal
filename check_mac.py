"""Check the destination environment without altering the trained artifacts."""
import platform
from pathlib import Path
import torch

print(f'Python: {platform.python_version()} | OS: {platform.system()} | architecture: {platform.machine()}')
print(f'PyTorch: {torch.__version__}')
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    print('Use a native arm64 Python installation to access Apple-silicon acceleration.')
if torch.backends.mps.is_available():
    probe = (torch.ones(4, device='mps') * 2).cpu()
    assert probe.tolist() == [2., 2., 2., 2.]
    print('MPS GPU check passed. The trainer still needs device support before using it.')
else:
    print('MPS is unavailable in this environment. The app can still run on CPU.')
root = Path(__file__).resolve().parent
for relative in ['artifacts/baseline.joblib', 'artifacts/distilbert_continued/best/model.safetensors', 'artifacts/distilbert_continued/training_checkpoint.pt']:
    print(f'{relative}: {"present" if (root / relative).is_file() else "MISSING"}')
