"""Real tiny PEFT/AdamW CPU recovery and negative observer cases."""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import save_file, load_file
import torch
from transformers.trainer import safe_globals

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vlm.runtime.resume_witness import ResumeWitness, rng_values, pinned_checkpoint
from vlm.runtime.resume_pair import compare_resume_pair


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        value = self.linear(x)
        return {'loss': value.square().mean()}


def create():
    torch.manual_seed(17)
    model = get_peft_model(Tiny(), LoraConfig(r=2, lora_alpha=4, lora_dropout=.05,
                                            target_modules=['linear']))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: (25 - step) / 25)
    return model, opt, scheduler


def update(model, optimizer, scheduler):
    for _ in range(4):
        (model(torch.tensor([[.2, -.3, .4]]))['loss'] / 4).backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def load_rng(checkpoint):
    with safe_globals():
        value = torch.load(Path(checkpoint) / 'rng_state.pth', weights_only=True)
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['cpu'])


class ResumeWitnessTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / 'checkpoint-20'
        self.checkpoint.mkdir()
        model, opt, scheduler = create()
        for _ in range(20):
            update(model, opt, scheduler)
        self.state = dict(global_step=20, max_steps=25, epoch=10.0, train_batch_size=1)
        self.args = dict(max_steps=25, per_device_train_batch_size=1,
                         gradient_accumulation_steps=4, resume_only_model=False,
                         ignore_data_skip=False, callbacks=['nvidia_baseline'])
        for name, value in [('trainer_state.json', self.state), ('args.json', self.args),
                            ('adapter_config.json', {'r': 2})]:
            (self.checkpoint / name).write_text(json.dumps(value), encoding='utf8')
        save_file(get_peft_model_state_dict(model), self.checkpoint / 'adapter_model.safetensors')
        torch.save(opt.state_dict(), self.checkpoint / 'optimizer.pt')
        torch.save(scheduler.state_dict(), self.checkpoint / 'scheduler.pt')
        torch.save(rng_values(), self.checkpoint / 'rng_state.pth')
        self.pins = {p.name: dict(bytes=p.stat().st_size, SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
                     for p in self.checkpoint.iterdir()}
        for _ in range(5):
            update(model, opt, scheduler)
        self.expected = (copy.deepcopy(get_peft_model_state_dict(model)),
                         copy.deepcopy(opt.state_dict()), copy.deepcopy(scheduler.state_dict()))
        self.model, self.opt, self.scheduler = create()
        set_peft_model_state_dict(self.model, load_file(self.checkpoint / 'adapter_model.safetensors'))
        self.opt.load_state_dict(torch.load(self.checkpoint / 'optimizer.pt', weights_only=True))
        self.scheduler.load_state_dict(torch.load(self.checkpoint / 'scheduler.pt', weights_only=True))
        self.trainer = SimpleNamespace(model=self.model, optimizer=self.opt,
            lr_scheduler=self.scheduler, state=SimpleNamespace(**self.state),
            args=SimpleNamespace(**self.args, resume_from_checkpoint=str(self.checkpoint)),
            _load_rng_state=load_rng)

    def witness(self, **kwargs):
        epoch = time.time()
        return ResumeWitness(self.trainer, self.model, range(8), self.root / 'witness',
            self.checkpoint, self.pins, rank=0, world_size=1, checkpoint_step=20,
            max_steps=25, epoch=epoch, deadline=epoch + 60, max_seconds=60, **kwargs)

    def test_opt_in_entry_state_is_captured_at_actual_forward_and_only_once(self):
        witness = self.witness(entry_state_step=20)
        self.train(witness)
        witness.checkpoint_saved()
        witness.finish()
        entry = json.loads((self.root / 'witness/forward-entry-state.json').read_bytes())
        result = json.loads((self.root / 'witness/result.json').read_bytes())
        self.assertEqual((entry['global_step'], entry['forward_ordinal']), (20, 1))
        self.assertTrue(entry['observer_RNG_unchanged'])
        self.assertTrue(result['entry_state_captured'])
        self.assertFalse(result['recovery_qualified'])

    def test_entry_state_outside_window_rejected(self):
        for step in (True, 19, 25, 30):
            with self.assertRaisesRegex(ValueError, 'entry-state step'):
                self.witness(entry_state_step=step)

    def train(self, witness, count=5):
        self.trainer._load_rng_state(self.checkpoint)
        for _ in range(count):
            update(self.model, self.opt, self.scheduler)
            self.trainer.state.global_step += 1
            witness.guarded(witness.step_end)()

    def test_real_cpu_restore_updates_and_observer_detaches(self):
        native = self.trainer._load_rng_state
        witness = self.witness()
        self.train(witness)
        witness.checkpoint_saved()
        witness.finish()
        self.assertIs(self.trainer._load_rng_state, native)
        for key, expected in self.expected[0].items():
            self.assertTrue(torch.equal(get_peft_model_state_dict(self.model)[key], expected))
        self.assertEqual(self.scheduler.state_dict(), self.expected[2])
        for key, state in self.expected[1]['state'].items():
            for name, expected in state.items():
                actual = self.opt.state_dict()['state'][key][name]
                self.assertTrue(torch.equal(actual, expected))
        result = json.loads((self.root / 'witness/result.json').read_bytes())
        self.assertEqual([x['global_step'] for x in result['steps']], [21, 22, 23, 24, 25])
        self.assertEqual(len(result['inputs']), 20)
        self.assertFalse(result['recovery_qualified'])

    def test_weights_without_optimizer_restore_rejected(self):
        self.trainer.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad])
        with self.assertRaises(ValueError):
            self.witness()
        self.assertTrue(json.loads((self.root / 'witness/progress.json').read_bytes())['failed'])

    def test_scheduler_restart_rejected(self):
        self.scheduler.last_epoch = 0
        with self.assertRaisesRegex(ValueError, 'values differ'):
            self.witness()

    def test_changed_adapter_rejected(self):
        with torch.no_grad():
            next(p for p in self.model.parameters() if p.requires_grad).add_(.01)
        with self.assertRaisesRegex(ValueError, 'values differ'):
            self.witness()

    def test_missing_rank_rng_pin_rejected(self):
        self.pins.pop('rng_state.pth')
        with self.assertRaisesRegex(ValueError, 'complete resume state'):
            self.witness()

    def test_silent_rng_loader_preserves_failure(self):
        self.trainer._load_rng_state = lambda checkpoint: None
        witness = self.witness()
        with self.assertRaisesRegex(ValueError, 'RNG differs'):
            self.trainer._load_rng_state(self.checkpoint)
        self.assertTrue(witness.failed)
        self.assertEqual(len(witness.handles), 0)

    def test_native_rng_loader_error_preserves_failure(self):
        def failed(checkpoint):
            raise RuntimeError('native fixture loader failed')
        self.trainer._load_rng_state = failed
        witness = self.witness()
        with self.assertRaisesRegex(RuntimeError, 'native fixture loader failed'):
            self.trainer._load_rng_state(self.checkpoint)
        self.assertTrue(witness.failed)
        self.assertIs(self.trainer._load_rng_state, failed)

    def test_forward_before_native_rng_restore_rejected(self):
        witness = self.witness()
        with self.assertRaisesRegex(ValueError, 'input or accumulation'):
            self.model(torch.tensor([[.2, -.3, .4]]))
        self.assertTrue(witness.failed)

    def test_partial_window_preserves_completed_denominator(self):
        witness = self.witness()
        self.train(witness, count=2)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            witness.guarded(witness.finish)()
        progress = json.loads((self.root / 'witness/progress.json').read_bytes())
        self.assertEqual(progress['completed_steps'], [21, 22])
        self.assertEqual(progress['inputs'], 8)
        self.assertTrue(progress['failed'])

    def test_checkpoint_callback_without_completed_update_rejected(self):
        witness = self.witness()
        self.trainer.state.global_step = 25
        with self.assertRaisesRegex(ValueError, 'checkpoint callback'):
            witness.guarded(witness.checkpoint_saved)()
        self.assertTrue(witness.failed)

    def test_intermediate_save_and_fresh_process_tail_match_real_values(self):
        args = dict(self.args, save_steps=10)
        def save(folder, trainer, step):
            folder.mkdir(parents=True)
            state = dict(global_step=step, max_steps=25, epoch=step / 2, train_batch_size=1)
            for name, value in [('trainer_state.json', state), ('args.json', args),
                                ('adapter_config.json', {'r': 2})]:
                (folder / name).write_text(json.dumps(value), encoding='utf8')
            save_file(get_peft_model_state_dict(trainer.model), folder / 'adapter_model.safetensors')
            torch.save(trainer.optimizer.state_dict(), folder / 'optimizer.pt')
            torch.save(trainer.lr_scheduler.state_dict(), folder / 'scheduler.pt')
            torch.save(rng_values(), folder / 'rng_state.pth')
            return {p.name: dict(bytes=p.stat().st_size, SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
                    for p in folder.iterdir()}
        model, opt, scheduler = create()
        for _ in range(10):
            update(model, opt, scheduler)
        initial = SimpleNamespace(model=model, optimizer=opt, lr_scheduler=scheduler)
        checkpoint10 = self.root / 'paired/checkpoint-10'
        pins10 = save(checkpoint10, initial, 10)
        def run(checkpoint, pins, start, directory):
            model, opt, scheduler = create()
            set_peft_model_state_dict(model, load_file(checkpoint / 'adapter_model.safetensors'))
            opt.load_state_dict(torch.load(checkpoint / 'optimizer.pt', weights_only=True))
            scheduler.load_state_dict(torch.load(checkpoint / 'scheduler.pt', weights_only=True))
            trainer = SimpleNamespace(model=model, optimizer=opt, lr_scheduler=scheduler,
                state=SimpleNamespace(**json.loads((checkpoint / 'trainer_state.json').read_bytes())),
                args=SimpleNamespace(**args, resume_from_checkpoint=str(checkpoint)),
                _load_rng_state=load_rng)
            epoch = time.time()
            witness = ResumeWitness(trainer, model, range(8), directory / 'witness', checkpoint,
                pins, rank=0, world_size=1, checkpoint_step=start, max_steps=25,
                epoch=epoch, deadline=epoch + 60, max_seconds=60)
            trainer._load_rng_state(checkpoint)
            emitted = {}
            for step in range(start + 1, 26):
                update(model, opt, scheduler)
                trainer.state.global_step = step
                witness.step_end()
                if step in (20, 25):
                    path = directory / ('checkpoint-%d' % step)
                    emitted[step] = (path, save(path, trainer, step))
                    witness.checkpoint_saved()
            witness.finish()
            return json.loads((directory / 'witness/result.json').read_bytes()), emitted
        continuous, control_files = run(checkpoint10, pins10, 10, self.root / 'paired/continuous')
        restored, resume_files = run(*control_files[20], 20, self.root / 'paired/restored')
        compared = compare_resume_pair(continuous, restored)
        self.assertEqual(compared['input_equal_forwards'], 20)
        self.assertEqual(compared['exact_loss_forwards'], 20)
        self.assertEqual(compared['exact_full_state_updates'], 5)
        control_final = pinned_checkpoint(*control_files[25], 1)
        restored_final = pinned_checkpoint(*resume_files[25], 1)
        for key in ('adapter', 'optimizer', 'scheduler', 'RNG'):
            self.assertEqual(control_final[key], restored_final[key])
        self.assertEqual(continuous['saved_checkpoints'], [20, 25])
        self.assertEqual(restored['saved_checkpoints'], [25])


if __name__ == '__main__':
    unittest.main()
