"""Atomic, trusted-local training checkpoints including RNG and optimizer state."""
import json
import os
import random
import numpy as np
import torch


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def capture_rng():
    state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, model, optimizer, scheduler, state):
    temporary = path.with_suffix('.tmp')
    payload = {'schema': 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rng': capture_rng(), 'state': state}
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model, optimizer, scheduler):
    # Only load checkpoints created by this project; torch pickle is not an interchange format.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['schema'] != 1:
        raise ValueError('Unsupported checkpoint schema')
    model.load_state_dict(payload['model'])
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    restore_rng(payload['rng'])
    return payload['state']
