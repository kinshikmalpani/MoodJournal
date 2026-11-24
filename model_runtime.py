"""Interchangeable local inference; never downloads weights from the app."""
import json
import threading
from pathlib import Path
import joblib
import numpy as np
from model_catalog import MODELS

ROOT = Path(__file__).resolve().parent
LABELS = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']


class TransformerPredictor:
    def __init__(self, folder):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        torch.set_num_threads(4)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(folder, local_files_only=True)
        self.model.eval()
        self.lock = threading.Lock()
        self.classes_ = np.array([self.model.config.id2label[i] for i in range(self.model.config.num_labels)])

    def predict_proba(self, texts):
        # Overflow windows retain every token even for unusually long individual words.
        output = []
        with self.lock, self.torch.inference_mode():
            for start in range(0, len(texts), 16):
                batch = list(texts[start:start + 16])
                encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_overflowing_tokens=True, return_tensors='pt')
                owners = encoded.pop('overflow_to_sample_mapping').numpy()
                chunks = []
                for offset in range(0, len(owners), 16):
                    inputs = {k: v[offset:offset + 16] for k, v in encoded.items()}
                    chunks.append(self.model(**inputs).logits.softmax(-1).numpy())
                probs = np.concatenate(chunks)
                output.extend(probs[owners == i].mean(axis=0) for i in range(len(batch)))
        return np.asarray(output)


def load_artifact(name='baseline'):
    if name == 'baseline':
        return joblib.load(ROOT / 'artifacts' / 'baseline.joblib')
    if name not in MODELS:
        raise ValueError('Unknown model')
    folder = ROOT / 'artifacts' / name
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    weights = ROOT / 'artifacts' / MODELS[name][2]
    return {'pipeline': TransformerPredictor(weights), 'threshold': metadata['threshold'], 'version': metadata['version']}
