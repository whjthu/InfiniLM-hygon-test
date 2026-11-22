"""Run rank-local generator and all-denominator bracket fixtures without Torch."""

from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_bracket as compare
import rng_treatment as trial


class ByteTensor(bytes):
    def detach(self):
        return self

    cpu = contiguous = detach

    def reshape(self, _):
        return self

    view = reshape
    numpy = detach


class Test(unittest.TestCase):
    def test_apply_actual_arm_logic_to_fake_eight_default_generators(self):
        for arm in trial.ARMS:
            with self.subTest(arm=arm):
                states = [ByteTensor(bytes([i])) for i in range(8)]
                sets = []

                def set_state(value, device):
                    sets.append(device)
                    states[device] = value

                rank = 3
                cuda = SimpleNamespace(current_device=lambda:rank, get_rng_state_all=lambda:list(states),
                                       set_rng_state=set_state)

                def private_generator(device):
                    self.assertEqual(device, ('cuda', rank))
                    device_seed = []
                    return SimpleNamespace(manual_seed=device_seed.append, get_state=lambda:
                                           ByteTensor(device_seed[0].to_bytes(8, 'little')))

                torch = SimpleNamespace(cuda=cuda, uint8='uint8', Generator=private_generator,
                                        device=lambda *value:value)

                def state(get):
                    return dict(python=17, numpy=23, cpu=ByteTensor(b'FAKE'), cuda=get())

                def fingerprint(value):
                    return sha256(repr(value).encode()).hexdigest()

                before = fingerprint(state(cuda.get_rng_state_all))
                result = trial.apply_rng(torch, state, fingerprint, before, rank, arm)
                self.assertEqual(sets, [rank])
                self.assertEqual(result['seed'], trial.SEED_BASE + rank if arm == 'b' else None)
                self.assertEqual(result['before']['non_device'], result['after']['non_device'])
                self.assertEqual(result['before']['full'] != result['after']['full'], arm == 'b')
                self.assertEqual([i for i in range(8) if states[i] != bytes([i])], [rank] if arm == 'b' else [])
                with self.assertRaises(ValueError):
                    trial.apply_rng(torch, state, fingerprint, 'foreign', rank, arm)

    def fixture(self, root, kind):
        step = '1103606.9999'
        history = {}
        trail = dict(config_SHA256='a' * 64, actual_step=step, epoch=10, deadline=730)
        for index, arm in enumerate(trial.ARMS):
            for rank in range(8):
                folder = root / arm / 'train/minicpm-usgs-witness' / f'rank-{rank:02d}'
                folder.mkdir(parents=True)
                actual_rng = ['original-device'] * 8
                historical = dict(full='archived', non_device='host', devices=actual_rng)
                seeded = dict(historical, devices=actual_rng[:])
                if arm == 'b':
                    seeded['full'] = 'new-rng'
                    seeded['devices'][rank] = 'changed-device'
                identity = dict(boot_id='boot', startticks=3, pid=100+8*index+rank, arm=arm, rank=rank, world_size=8,
                                checkpoint_step=20, actual_step=step, config_SHA256=trail['config_SHA256'])
                restored = dict(adapter='a', optimizer='b', scheduler='c')
                history[rank] = dict(input='historical-input', state=restored, RNG='archived')
                loss = 1.0 + float(arm == 'b')
                if kind == 'A_DIVERGENCE' and arm == 'a2' and rank == 0:
                    loss += 0.25
                post = dict(rank=rank, state='ALL_RANKS_STOPPED', global_step=20, backward_calls=0, optimizer_updates=0)
                observe = dict(post, world_size=8, checkpoint_step=20, state='STOP_BEFORE_BACKWARD',
                               planned_forwards=1, completed_forwards=1, RNG_after_native_restore='archived',
                               RNG_before_compute_loss=seeded['full'], operators=compare.KNOWN_ROUTE, ddp='same',
                               language_before_forward='same', compute_loss_source='same', inner_linear='same',
                               RNG_after_forward=seeded['full']+'+work', restored_state=restored,
                               root_before_forward=dict(RNG=seeded['full'], context='same'),
                               actual_input=dict(ordinal=1, global_step=20, full_input='historical-input'),
                               actual_loss=dict(ordinal=1, loss=hex(rank)+str(loss), loss_scalar=loss))
                applied = dict(run=identity, treatment=dict(arm=arm, rank=rank, before=historical, after=seeded,
                               requested_device_SHA256=seeded['devices'][rank],
                               seed=trial.SEED_BASE+rank if arm == 'b' else None))
                audit = dict(run=identity, training_modes_unchanged=True, before='same', after='same',
                             opaque_cache_attrs_before=[], opaque_cache_attrs_after=[],
                             observed_buffer_rotary_changed=False, parameter_identity_and_versions_unchanged=True,
                             root_input_identity_and_versions_unchanged=True, backward_calls=0, optimizer_updates=0)
                ready = dict(run=trail, state='FORWARD_RETURNED', rank=rank, world_size=8, checkpoint_step=20)
                if kind == 'FOREIGN_RNG' and arm == 'b' and rank == 0:
                    seeded['devices'][1] = 'foreign change'
                if kind == 'FOREIGN_EPOCH' and arm == 'b' and rank == 0:
                    ready['run'] = dict(trail, epoch=20)
                for name, value in (('first-forward-state', observe), ('first-forward-stop', post),
                                    ('first-forward-ready', ready), ('rng-treatment', applied), ('rng-state-audit', audit)):
                    if kind == 'MISSING' and arm == 'b' and rank == 3 and name == 'rng-state-audit':
                        continue
                    (folder / (name+'.json')).write_text(json.dumps(value), encoding='utf-8')
        return step, history

    def test_complete_and_negative_evidence_denominators(self):
        original = Path(__file__).resolve().parents[3] / (
            'results/bw1000/2026-11-22/training-minicpm-original1189-first-forward-SDPA-route.json')
        expected = {'GOOD': (24, 8, 'OBSERVED_RNG_SENSITIVITY'), 'A_DIVERGENCE': (24, 8, 'CONTROL_REPEAT_DIFFERS'),
                    'FOREIGN_RNG': (23, 7, 'INCOMPLETE'), 'FOREIGN_EPOCH': (24, 0, 'INCOMPLETE'),
                    'MISSING': (23, 7, 'INCOMPLETE')}
        for case, outcome in expected.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as folder:
                root = Path(folder).resolve(strict=True)
                step, history = self.fixture(root, case)
                result = compare.consume(root, step, original, history)
                self.assertEqual((result['verified_individual_forwards'], result['comparable_ranks'],
                                  result['interpretation']), outcome)
                self.assertEqual(result['planned_forwards'], 24)
                self.assertIs(result['historical_cause_proven'], False)
                self.assertIs(result['GPU_nondeterminism_proven'], False)


if __name__ == '__main__':
    unittest.main()
