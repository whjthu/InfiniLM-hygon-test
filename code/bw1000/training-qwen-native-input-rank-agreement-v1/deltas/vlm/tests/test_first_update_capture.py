"""Real PEFT/AdamW CPU fixtures; no original weights, GPU or numerical qualification."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vlm.runtime.first_update_capture import FirstUpdateCapture, remaining
from vlm.runtime.first_update_audit import fingerprint


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = get_peft_model(torch.nn.Sequential(torch.nn.Linear(7, 9), torch.nn.Tanh(), torch.nn.Linear(9, 4)),
            LoraConfig(r=2, lora_alpha=4, lora_dropout=.05, target_modules=['0', '2']))

    def forward(self, x):
        logits = self.adapter(x)
        return dict(logits=logits, loss=100*logits.square().mean())


class FirstUpdateCaptureTest(unittest.TestCase):
    def fixture(self, directory, **kwargs):
        torch.manual_seed(42)
        model = TinyModel()
        named = sorted((n, p) for n, p in model.named_parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW([p for _, p in named], lr=0, weight_decay=.1)
        trainer = SimpleNamespace(model=model, optimizer=optimizer, state=SimpleNamespace(global_step=0),
                                  args=SimpleNamespace(max_grad_norm=1.0))
        def clip(wrapped):
            return torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0)
        trainer._clip_grad_norm = clip
        epoch = time.time()
        capture = FirstUpdateCapture(trainer, Path(directory)/'rank-00', named, rank=0, world_size=1,
            epoch=epoch, deadline=epoch+90, max_seconds=90,
            rng=lambda: dict(torch_CPU=torch.get_rng_state().clone()), **kwargs)
        return model, trainer, capture, clip

    def test_genuine_two_updates_full_artifacts_clipping_and_native25(self):
        with tempfile.TemporaryDirectory() as d:
            model, trainer, capture, original_clip = self.fixture(d)
            frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
            capture.guarded(capture.attach)(model)
            for step in range(25):
                trainer.optimizer.zero_grad(set_to_none=True)
                for micro in range(4):
                    result = model(torch.arange(21).float().reshape(3, 7)/20 + micro/10)
                    (result['loss']/4).backward()
                trainer._clip_grad_norm(model)
                capture.guarded(capture.before_optimizer)()
                trainer.optimizer.step()
                capture.guarded(capture.after_optimizer)()
                trainer.state.global_step += 1
                capture.guarded(capture.step_end)()
                for group in trainer.optimizer.param_groups:
                    group['lr'] = .01
            capture.guarded(capture.train_end)()
            self.assertIs(trainer._clip_grad_norm, original_clip)
            self.assertEqual((capture.inputs, capture.outputs, capture.updates, capture.clip_calls), (8, 8, 2, 2))
            self.assertEqual(len(capture.files), 25)
            values = {}
            for row in capture.files:
                path = capture.root/row['name']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row['SHA256'])
                self.assertEqual(path.stat().st_size, row['bytes'])
                values[row['name']] = torch.load(path, map_location='cpu', weights_only=True)
            first_before = values['update-01-before-optimizer.pt']['parameters']
            first_after = values['update-01-after-optimizer.pt']['parameters']
            self.assertTrue(all(torch.equal(v, first_after[n]) for n, v in first_before.items()))
            second_before = values['update-02-before-optimizer.pt']['parameters']
            second_after = values['update-02-after-optimizer.pt']['parameters']
            self.assertTrue(any(not torch.equal(v, second_after[n]) for n, v in second_before.items()))
            self.assertTrue(all(v for v in values['update-02-after-optimizer.pt']['optimizer']['state_by_name'].values()))
            pre = values['update-02-preclip.pt']['gradients']
            post = values['update-02-postclip.pt']['gradients']
            self.assertTrue(any(not torch.equal(v, post[n]) for n, v in pre.items() if v is not None))
            for n, p in model.named_parameters():
                if n in frozen:
                    self.assertTrue(torch.equal(p, frozen[n]))
            self.assertEqual(json.loads((capture.root/'native25-complete.json').read_text())['global_step'], 25)
            self.assertFalse(torch.cuda.is_initialized())

    def test_pinned_microbatch_mismatch_stops_before_model_forward(self):
        with tempfile.TemporaryDirectory() as d:
            tensor = torch.arange(21).float().reshape(3, 7)
            pin = fingerprint(dict(args=(tensor,), kwargs={}))['SHA256']
            model, trainer, capture, original_clip = self.fixture(d, input_fingerprints=[pin] * 8)
            capture.guarded(capture.attach)(model)
            with patch.object(model, 'forward', wraps=model.forward) as forward:
                model(tensor)
                with self.assertRaisesRegex(ValueError, 'rank 0, microbatch 2'):
                    model(tensor + 1)
                self.assertEqual(forward.call_count, 1)
            self.assertEqual((capture.inputs, capture.outputs, capture.updates), (1, 1, 0))
            self.assertEqual(len(trainer.optimizer.state), 0)
            self.assertIs(trainer._clip_grad_norm, original_clip)
            self.assertFalse((capture.root / 'forward-02-input.pt').exists())
            failure = json.loads((capture.root / 'failure.json').read_text())
            self.assertEqual(failure['planned_micro_forwards'], 8)

    def test_peer_rejection_stops_a_locally_matching_batch_before_forward(self):
        with tempfile.TemporaryDirectory() as d:
            tensor = torch.arange(21).float().reshape(3, 7)
            pin = fingerprint(dict(args=(tensor,), kwargs={}))['SHA256']
            votes = []

            def agreement(matched):
                votes.append(matched)
                return False

            model, trainer, capture, original_clip = self.fixture(
                d, input_fingerprints=[pin] * 8, input_agreement=agreement)
            capture.guarded(capture.attach)(model)
            with patch.object(model, 'forward', wraps=model.forward) as forward:
                with self.assertRaisesRegex(ValueError, 'peer rank at microbatch 1'):
                    model(tensor)
                forward.assert_not_called()
            self.assertEqual(votes, [True])
            self.assertEqual((capture.inputs, capture.outputs, capture.updates), (0, 0, 0))
            self.assertEqual(len(trainer.optimizer.state), 0)
            self.assertIs(trainer._clip_grad_norm, original_clip)

    def test_capture_failure_keeps_denominator_and_restores_original_clip(self):
        with tempfile.TemporaryDirectory() as d:
            model, trainer, capture, original_clip = self.fixture(d)
            capture.guarded(capture.attach)(model)
            with self.assertRaises(RuntimeError):
                model(torch.full((3, 7), float('nan')))
            self.assertIs(trainer._clip_grad_norm, original_clip)
            row = json.loads((capture.root/'failure.json').read_text())
            self.assertEqual((row['inputs_saved'], row['outputs_saved'], row['completed_updates']), (0, 0, 0))
            self.assertEqual((row['planned_micro_forwards'], row['planned_updates'], row['planned_native_global_step']), (8, 2, 25))
            self.assertFalse((capture.root/'receipt.json').exists())
            self.assertFalse((capture.root/'native25-complete.json').exists())

    def test_partial_budget_and_incomplete_native_run_never_pass(self):
        with tempfile.TemporaryDirectory() as d:
            model, trainer, capture, original_clip = self.fixture(d, artifact_limit=100)
            with self.assertRaises((ValueError, RuntimeError)):
                capture.guarded(capture.attach)(model)
            self.assertTrue(capture.failed)
            self.assertEqual(capture.files, [])
            self.assertIs(trainer._clip_grad_norm, original_clip)
        with tempfile.TemporaryDirectory() as d:
            model, _, capture, _ = self.fixture(d)
            capture.guarded(capture.attach)(model)
            with self.assertRaises(ValueError):
                capture.guarded(capture.train_end)()
            self.assertFalse((capture.root/'native25-complete.json').exists())

    def test_deadline_rejects_invalid_clocks_and_never_renews_epoch(self):
        self.assertEqual(remaining(100, 120, 110, 20), 10)
        for args in ((100, 120, 120, 20), (100, 120, 99, 20), (100, 121, 110, 20),
                     (0, 20, 1, 20), (100, 120, float('nan'), 20), (100, 120, 110, 21601)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                remaining(*args)
        with tempfile.TemporaryDirectory() as d:
            _, _, capture, _ = self.fixture(d)
            with patch('vlm.runtime.first_update_capture.time.monotonic', return_value=capture.anchor+91):
                with self.assertRaises(TimeoutError):
                    capture.check_deadline()


if __name__ == '__main__':
    unittest.main()
