"""Real PEFT/AdamW CPU training checks for ordered native dropout capture and replay."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from peft import LoraConfig, get_peft_model
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vlm.runtime.lora_dropout_session import LoRADropoutSession, input_identity


def model():
    torch.manual_seed(41)
    result = get_peft_model(torch.nn.Sequential(torch.nn.Linear(7, 9), torch.nn.Tanh(), torch.nn.Linear(9, 4)),
                            LoraConfig(r=2, lora_alpha=4, lora_dropout=.05, target_modules=['0', '2']))
    with torch.no_grad():
        for name, value in result.named_parameters():
            if '.lora_B.' in name:
                value.copy_(torch.arange(value.numel()).reshape_as(value).float() / 1000)
    result.train()
    return result


def features(value):
    return {name.replace('.lora_A.default.weight', '.lora_dropout.default'): p.shape[-1]
            for name, p in value.named_parameters() if '.lora_A.default.weight' in name}


def tensors(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: tensors(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensors(v) for v in value]
    return value


def train(value, session=None, seed=731):
    opt = torch.optim.AdamW([p for p in value.parameters() if p.requires_grad], lr=.001)
    torch.manual_seed(seed); rows = []
    for ordinal in range(1, 9):
        x = (torch.arange(21).reshape(3, 7).float() - ordinal) / 40
        if session is None:
            output = value(x)
        else:
            with session.forward(global_step=(ordinal - 1) // 4, micro_ordinal=ordinal,
                                 input_SHA256=input_identity((x,), {})):
                output = value(x)
        loss = output.square().mean() / 4; loss.backward()
        row = {'output': output.detach().clone(), 'loss': loss.detach().clone(), 'rng': torch.get_rng_state().clone()}
        if ordinal % 4 == 0:
            row['preclip'] = {n: p.grad.detach().clone() for n, p in value.named_parameters() if p.requires_grad}
            row['norm'] = torch.nn.utils.clip_grad_norm_([p for p in value.parameters() if p.requires_grad], 1.0)
            row['postclip'] = {n: p.grad.detach().clone() for n, p in value.named_parameters() if p.requires_grad}
            opt.param_groups[0]['lr'] = 0 if ordinal == 4 else .001
            opt.step(); row['optimizer'] = tensors(opt.state_dict())
            row['parameters'] = {n: p.detach().clone() for n, p in value.named_parameters() if p.requires_grad}
            opt.zero_grad()
        rows.append(row)
    if session is not None:
        session.close()
    return rows


class LoRADropoutSessionTest(unittest.TestCase):
    def equal(self, a, b, *, rng=True):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b)); return
        if isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for k in a:
                if k != 'rng' or rng:
                    self.equal(a[k], b[k], rng=rng)
        elif isinstance(a, list):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):self.equal(x, y, rng=rng)
        else:self.assertEqual(a, b)

    def test_real_peft_two_updates_capture_and_other_seed_replay(self):
        original = model(); capture = copy.deepcopy(original); replay = copy.deepcopy(original)
        initial = {n: p.detach().clone() for n, p in original.named_parameters() if p.requires_grad}
        baseline = train(original)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); args = dict(binding={'recipe': 'test-p05-original', 'initial': 'frozen'}, rank=0, world_size=1)
            session = LoRADropoutSession(capture, root/'capture', mode='capture', expected_input_features=features(capture), **args)
            observed = train(capture, session)
            self.equal(observed, baseline)
            manifest_pin = hashlib.sha256((root/'capture/complete.json').read_bytes()).hexdigest()
            session = LoRADropoutSession(replay, root/'replay', mode='replay', reference=root/'capture',
                                        reference_manifest_SHA256=manifest_pin, expected_input_features=features(replay), **args)
            replayed = train(replay, session, seed=919)
            self.equal(replayed, baseline, rng=False)
            native_other_rng = train(model(), seed=919)
            for actual, native in zip(replayed, native_other_rng):self.equal(actual['rng'], native['rng'])
            self.equal(observed[3]['parameters'], initial)
            self.assertTrue(any(not torch.equal(initial[n], observed[7]['parameters'][n]) for n in initial))
            self.assertTrue(all(int(v['step']) == 2 for v in observed[7]['optimizer']['state'].values()))
            self.assertTrue(all('forward' not in m.__dict__ for n, m in capture.named_modules() if '.lora_dropout.' in n))
            self.assertEqual(json.loads((root/'capture/complete.json').read_bytes())['native_dropout_calls'], 16)

    def test_incomplete_forward_restores_native_modules_and_keeps_denominator(self):
        value = model()
        with tempfile.TemporaryDirectory() as temp:
            session = LoRADropoutSession(value, Path(temp)/'capture', mode='capture', binding={}, rank=0, world_size=1)
            x = torch.ones(3, 7)
            with session.forward(global_step=0, micro_ordinal=1, input_SHA256=input_identity((x,), {})):value(x)
            with self.assertRaises(ValueError):session.close()
            progress = json.loads((Path(temp)/'capture/progress.json').read_bytes())
            self.assertEqual(progress['completed_forwards'], 1); self.assertEqual(progress['planned_forwards'], 8)
            self.assertIsNotNone(progress['failure']); self.assertFalse((Path(temp)/'capture/complete.json').exists())
            self.assertTrue(all('forward' not in m.__dict__ for n, m in value.named_modules() if '.lora_dropout.' in n))

    def test_wrong_update_and_mask_budget_reject_without_rng_draw(self):
        for kind in ('ordinal', 'update', 'input_identity', 'budget', 'width'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                value = model(); x = torch.ones(3, 7); widths = features(value)
                if kind == 'width':widths = {n: 999 for n in widths}
                session = LoRADropoutSession(value, Path(temp)/'capture', mode='capture', binding={}, rank=0, world_size=1,
                                            expected_input_features=widths, byte_budget=1 if kind=='budget' else 1024**3)
                rng = torch.get_rng_state().clone()
                with self.assertRaises((ValueError, RuntimeError)):
                    with session.forward(global_step=1 if kind=='update' else 0, micro_ordinal=2 if kind=='ordinal' else 1,
                                         input_SHA256='bad' if kind=='input_identity' else input_identity((x,), {})):value(x)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(json.loads((Path(temp)/'capture/progress.json').read_bytes())['completed_forwards'], 0)

    def test_replay_rejects_wrong_model_input_call_order_and_corrupt_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); value = model(); session = LoRADropoutSession(value, root/'capture', mode='capture', binding={}, rank=0, world_size=1)
            train(value, session); pin = hashlib.sha256((root/'capture/complete.json').read_bytes()).hexdigest()
            for kind in ('manifest', 'input', 'order', 'bytes'):
                with self.subTest(kind=kind):
                    value = model(); rng = torch.get_rng_state().clone()
                    if kind == 'manifest':
                        with self.assertRaises(ValueError):LoRADropoutSession(value, root/kind, mode='replay', binding={}, rank=0, world_size=1, reference=root/'capture', reference_manifest_SHA256='0'*64)
                    else:
                        session = LoRADropoutSession(value, root/kind, mode='replay', binding={}, rank=0, world_size=1, reference=root/'capture', reference_manifest_SHA256=pin)
                        x = (torch.arange(21).reshape(3, 7).float() - 1) / 40
                        if kind == 'bytes':
                            path = root/'capture/forward-0001-masks.pt'; data=path.read_bytes(); path.write_bytes(data[:-1]+bytes([data[-1]^1]))
                        with self.assertRaises(ValueError):
                            with session.forward(global_step=0, micro_ordinal=1, input_SHA256='a'*64 if kind=='input' else input_identity((x,), {})):
                                if kind=='order':value.get_submodule('base_model.model.2.lora_dropout.default')(torch.ones(3, 9))
                                else:value(x)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_input_identity_covers_labels_pixels_dtype_shape_and_scalar(self):
        a = input_identity((), {'input_ids': torch.tensor([1, 2]), 'labels': torch.tensor([3]), 'pixel_values': torch.ones(2, 3), 'flag': torch.tensor(1)})
        b = input_identity((), {'flag': torch.tensor(1), 'pixel_values': torch.ones(2, 3), 'labels': torch.tensor([3]), 'input_ids': torch.tensor([1, 2])})
        self.assertEqual(a, b)
        for key in ('input_ids', 'labels', 'pixel_values'):
            row = {'input_ids': torch.tensor([1, 2]), 'labels': torch.tensor([3]), 'pixel_values': torch.ones(2, 3), 'flag': torch.tensor(1)}
            row[key] = row[key] + 1
            self.assertNotEqual(a, input_identity((), row))

if __name__ == '__main__':unittest.main()
