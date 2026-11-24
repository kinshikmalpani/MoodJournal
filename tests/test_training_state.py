import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from training_state import save_checkpoint, load_checkpoint


class ResumeTests(unittest.TestCase):
    def test_resume_reproduces_next_optimizer_update_and_rng(self):
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        def setup():
            model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(.3), torch.nn.Linear(5, 2))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=0)
            return model, optimizer, scheduler
        model, optimizer, scheduler = setup()
        data = torch.ones(4, 3)
        target = torch.tensor([0, 1, 0, 1])
        def step(model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()
        step(model, optimizer)
        scheduler.step(.5)
        scheduler.step(.4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.pt'
            state = {'epoch': 1, 'next_batch': 7, 'batch_order': [[1, 3], [2, 0]]}
            save_checkpoint(path, model, optimizer, scheduler, state)
            expected_rng = (random.random(), np.random.rand())
            step(model, optimizer)
            expected_weights = {k: v.clone() for k, v in model.state_dict().items()}
            restored, opt2, sch2 = setup()
            loaded = load_checkpoint(path, restored, opt2, sch2)
            self.assertEqual(loaded, state)
            self.assertEqual((random.random(), np.random.rand()), expected_rng)
            self.assertEqual(scheduler.state_dict(), sch2.state_dict())
            self.assertEqual(optimizer.param_groups[0]['lr'], opt2.param_groups[0]['lr'])
            step(restored, opt2)
            for key, expected in expected_weights.items():
                torch.testing.assert_close(restored.state_dict()[key], expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
