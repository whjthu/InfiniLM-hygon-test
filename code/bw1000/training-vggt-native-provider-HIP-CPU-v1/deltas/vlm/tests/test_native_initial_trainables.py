"""CPU callback orchestration; model-contract/device qualification is separate."""
import hashlib
import json
import os
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
from vlm.plugins import native_initial_trainables as plugin
from vlm.runtime.initial_trainables import named_parameter_sha256


class NativeFirstUpdates:
    pass


class BaselineCallback:
    pass


class NativeInitializationTest(unittest.TestCase):
    def fixture(self, root):
        def model(seed):
            torch.manual_seed(seed)
            return get_peft_model(torch.nn.Sequential(torch.nn.Linear(7, 9)),
                                  LoraConfig(r=2, lora_alpha=4, target_modules=['0']))
        source, target = model(41), model(876)
        named = sorted((n, p) for n, p in target.named_parameters() if p.requires_grad)
        reference = sorted((n, p) for n, p in source.named_parameters() if p.requires_grad)
        original = dict(parameters={n: p.detach().clone() for n, p in reference},
                        manifest=[dict(name=n, shape=list(p.shape), dtype=str(p.dtype), numel=p.numel())
                                  for n, p in reference], global_step=0, RNG={'unused': True})
        artifact = root / 'initial.pt'
        torch.save(original, artifact)
        binding = dict(initial_state=dict(path=str(artifact), bytes=artifact.stat().st_size,
                       SHA256=hashlib.sha256(artifact.read_bytes()).hexdigest()),
                       initial_trainables_SHA256=named_parameter_sha256(reference))
        callback = object.__new__(plugin.NativeInitialTrainables)
        callback.trainer = SimpleNamespace(model=target,
            optimizer=torch.optim.AdamW([p for _, p in named]),
            callback_handler=SimpleNamespace(callbacks=[callback, NativeFirstUpdates(), BaselineCallback()]))
        epoch = time.time()
        env = dict(FIRST_UPDATE_EPOCH=str(epoch), FIRST_UPDATE_DEADLINE=str(epoch + 60),
                   FIRST_UPDATE_MAX_SECONDS='60', FIRST_UPDATE_BINDING=str(root / 'binding.json'),
                   FIRST_UPDATE_CAPTURE=str(root / 'capture'), RANK='0', WORLD_SIZE='8')
        return callback, named, binding, env

    def invoke(self, callback, named, binding, env, **changes):
        raw = json.dumps(binding).encode()
        Path(env['FIRST_UPDATE_BINDING']).write_bytes(raw)
        env = dict(env, FIRST_UPDATE_BINDING_SHA256=hashlib.sha256(raw).hexdigest(), **changes)
        with patch.dict(os.environ, env), patch.object(plugin, 'validate_native_recipe', return_value=(named, {})):
            return callback.on_train_begin(None, SimpleNamespace(global_step=0), 'control')

    def test_actual_cpu_copy_preserves_optimizer_rng_and_capture_namespace(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            callback, named, binding, env = self.fixture(root)
            objects = [id(p) for _, p in named]
            before_rng = torch.get_rng_state().clone()
            self.assertEqual(self.invoke(callback, named, binding, env), 'control')
            self.assertEqual(named_parameter_sha256(named), binding['initial_trainables_SHA256'])
            self.assertEqual(objects, [id(p) for _, p in named])
            self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
            self.assertEqual(len(callback.trainer.optimizer.state), 0)
            self.assertFalse((root / 'capture').exists())
            receipt = json.loads((root / 'capture-initialization/rank-00.json').read_bytes())
            self.assertTrue(receipt['passed'])
            self.assertFalse(receipt['RNG_restored'])
            self.assertFalse(receipt['numerical_alignment_qualified'])
            with self.assertRaises(FileExistsError):
                self.invoke(callback, named, binding, env)

    def test_bad_order_epoch_world_and_named_pin_do_not_modify_parameters(self):
        for bad in ('order', 'epoch', 'world', 'named_pin', 'file_pin', 'schema'):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                callback, named, binding, env = self.fixture(root)
                before = named_parameter_sha256(named)
                if bad == 'order':
                    callback.trainer.callback_handler.callbacks.reverse()
                if bad == 'epoch':
                    env['FIRST_UPDATE_DEADLINE'] = str(time.time() - 1)
                if bad == 'world':
                    env['WORLD_SIZE'] = '1'
                if bad == 'named_pin':
                    binding['initial_trainables_SHA256'] = '0' * 64
                if bad == 'file_pin':
                    binding['initial_state']['SHA256'] = '0' * 64
                if bad == 'schema':
                    binding['initial_state']['foreign'] = True
                with self.assertRaises(ValueError):
                    self.invoke(callback, named, binding, env)
                self.assertEqual(named_parameter_sha256(named), before)
                receipt = root / 'capture-initialization/rank-00.json'
                if receipt.exists():
                    self.assertFalse(json.loads(receipt.read_bytes())['passed'])


if __name__ == '__main__':
    unittest.main()
