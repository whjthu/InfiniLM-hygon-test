"""Actual CPU AdamW capture and frozen native math-path checks."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gencuda.vggt_first_update import FirstUpdateRecorder, create_vggt_recorder


class StripRecorder(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None if node.module == 'vggt_first_update' else node

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id == 'first_update_recorder' for t in node.targets):
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        if isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name):
            if node.test.left.id in ('first_update_recorder', 'observer'):
                return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == '_camera_loss':
            node.keywords = [k for k in node.keywords if k.arg != 'observer']
        return node

    def visit_FunctionDef(self, node):
        if node.name == 'train_step' and any(isinstance(n, ast.Attribute) and n.attr == 'guard'
                                             for n in ast.walk(node)):
            return None
        if node.name == '_native_train_step':
            node.name = 'train_step'
        if node.name == '_camera_loss':
            assert node.args.args[-1].arg == 'observer'
            node.args.args.pop()
            node.args.defaults.pop()
        return self.generic_visit(node)


class VGGTFirstUpdateTest(unittest.TestCase):
    def fixture(self, root, **limits):
        torch.manual_seed(193)
        model = torch.nn.Linear(4, 9)
        named = list(model.named_parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
        epoch = time.time()
        capture = FirstUpdateRecorder(root, named, optimizer, epoch=epoch, deadline=epoch + 60,
                                      device_ids=[], binding={'CPU_fixture': True}, **limits)
        return model, optimizer, capture

    def step(self, model, optimizer, capture, ordinal, mutation=None):
        batch = {'images': torch.arange(8).float().reshape(1, 2, 4) / 8}
        capture.enter(ordinal, batch)
        optimizer.zero_grad(set_to_none=True)
        value = model(batch['images'])
        target = torch.ones_like(value)
        capture.targets(target, torch.ones(1, 2, dtype=torch.bool))
        if mutation:
            mutation(batch)
        loss = (value - target).square().mean()
        capture.forward({'pose_enc_list': [value, value, value, value]}, loss)
        loss.backward()
        capture.gradients('preclip')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .05)
        capture.gradients('postclip', norm)
        optimizer.step()
        capture.updated()

    def test_native25_real_cpu_updates_typed_files_and_first_update_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'capture'
            model, optimizer, capture = self.fixture(root)
            for ordinal in range(1, 26):
                capture.guard(self.step, model, optimizer, capture, ordinal)
            capture.guard(capture.finish, 25, True)
            self.assertEqual(len(capture.files), 5 * 2 + 6)
            self.assertEqual(capture.completed_steps, 25)
            self.assertEqual(capture.phase, 'NATIVE25_RETURNED')
            self.assertEqual(len(optimizer.state), 2)
            for row in capture.files:
                raw = (root / row['name']).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), row['SHA256'])
                self.assertEqual(len(raw), row['bytes'])
                torch.load(root / row['name'], map_location='cpu', weights_only=True)
            initial = torch.load(root / 'initial/0000.pt', weights_only=True)['value']
            updated = torch.load(root / 'updated/0000.pt', weights_only=True)['value']
            self.assertFalse(torch.equal(initial, updated))
            pre = torch.load(root / 'preclip/0000.pt', weights_only=True)['value']
            post = torch.load(root / 'postclip/0000.pt', weights_only=True)['value']
            self.assertLess(float(post.norm()), float(pre.norm()))
            state = torch.load(root / 'optimizer/0000.pt', weights_only=True)['state']
            self.assertEqual(float(state['step']), 1)
            self.assertEqual(float(next(iter(optimizer.state.values()))['step']), 25)

    def test_initial_capture_does_not_change_rng_or_parameters(self):
        with tempfile.TemporaryDirectory() as d:
            model, _optimizer, capture = self.fixture(Path(d) / 'capture')
            before = [p.detach().clone() for p in model.parameters()]
            rng = torch.get_rng_state().clone()
            capture.guard(capture.enter, 1, {'images': torch.ones(1, 2, 4)})
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(p, v) for p, v in zip(model.parameters(), before)))

    def test_changed_input_preserves_partial_files_and_failure_denominator(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'capture'
            model, optimizer, capture = self.fixture(root)
            with self.assertRaises(ValueError):
                capture.guard(self.step, model, optimizer, capture, 1,
                              lambda batch: batch['images'].add_(1))
            progress = json.loads((root / 'progress.json').read_bytes())
            self.assertTrue(progress['failed'])
            self.assertEqual(progress['planned_global_steps'], 25)
            self.assertEqual(progress['completed_global_steps'], 0)
            self.assertGreater(len(progress['files']), 0)

    def test_partial_schedule_and_changed_frozen_boundary_cannot_finish(self):
        for bad in ('partial', 'frozen'):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as d:
                model, optimizer, capture = self.fixture(Path(d) / 'capture')
                last = 24 if bad == 'partial' else 25
                for ordinal in range(1, last + 1):
                    capture.guard(self.step, model, optimizer, capture, ordinal)
                with self.assertRaises(ValueError):
                    capture.guard(capture.finish, last, bad != 'frozen')
                self.assertTrue(capture.failed)
                self.assertEqual(capture.completed_steps, last)

    def test_out_of_order_and_budget_failures_do_not_accept_an_update(self):
        with tempfile.TemporaryDirectory() as d:
            model, _optimizer, capture = self.fixture(Path(d) / 'capture')
            with self.assertRaises(ValueError):
                capture.guard(capture.enter, 2, {})
            self.assertTrue(capture.failed)
        with tempfile.TemporaryDirectory() as d:
            _model, _optimizer, capture = self.fixture(Path(d) / 'capture', artifact_limit=16)
            with self.assertRaises(Exception):
                capture.guard(capture.enter, 1, {})
            self.assertTrue(capture.failed)
            self.assertFalse(capture.first_complete)

    def test_default_off_has_no_model_or_environment_side_effect(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(create_vggt_recorder(None, None, None, None, None))

    def test_stripped_backend_matches_frozen_native_math_ast(self):
        path = Path(__file__).resolve().parents[1] / 'src/gencuda/backends/vggt_training.py'
        tree = StripRecorder().visit(ast.parse(path.read_bytes()))
        actual = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
        self.assertEqual(actual, 'f581fd5021d21e7531b29aad55088444c7fb3f5cbc8152c42b536567892b4ff6')


if __name__ == '__main__':
    unittest.main()
