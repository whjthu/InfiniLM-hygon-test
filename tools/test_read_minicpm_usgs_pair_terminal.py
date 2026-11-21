from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import read_minicpm_usgs_pair_terminal as reader


class OriginalMiniCPMTest(unittest.TestCase):
    def setUp(self):
        scope = tempfile.TemporaryDirectory()
        self.addCleanup(scope.cleanup)
        self.root = Path(scope.name).resolve()
        self.publication = 'a' * 40
        self.epoch = 1000.0
        self.rows = {phase: [reader.PARENT + f'.{100 + i}', reader.LABEL+'-'+phase, 'COMPLETED', '0:0', '12',
              reader.NODE, resources] for i, (phase, resources) in enumerate({
                  'pre': 'cpu=2,node=1,mem=2G', 'gpu': 'cpu=64,node=1,mem=224000M,gres/dcu=8',
                  'post': 'cpu=2,node=1,mem=2G'}.items())}
        self.pci = [f'0000:{i:02x}:00.0' for i in range(4, 12)]
        self.intent = self.write('training-intent.json', dict(request=reader.REQUEST, publication=self.publication,
            parent=reader.PARENT, node=reader.NODE, replay_allowed=False, epoch=self.epoch,
            deadline=self.epoch+2400, native_deadline=self.epoch+1980))['SHA256']

    def write(self, name, value):
        path = self.root / name
        raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return dict(bytes=len(raw), SHA256=sha256(raw).hexdigest())

    def accounting(self):
        return ('\n'.join('|'.join(row) for row in self.rows.values())+'\n').encode()

    def audit(self):
        return reader.audit(self.root, self.accounting(), self.publication, self.intent)

    def materialize(self):
        phases = {}
        for phase, row in self.rows.items():
            phases[phase] = dict(epoch=self.epoch, row=row, terminal_visible=True, passed=True)
            self.write(phase+'-accounting.json', phases[phase])
        self.write('training-outcome.json', dict(request=reader.REQUEST, publication=self.publication,
            epoch=self.epoch, completed=True, clean_post=True, phases=phases))
        node = dict(node=reader.NODE, pci=self.pci, full_fd_inventory_performed=True, kfd_holder_count=0,
                    boot_id='boot-fixture', master_port=reader.PORT)
        self.write('before.json', dict(node, step=self.rows['pre'][0], monotonic_ns=100))
        self.write('after.json', dict(node, step=self.rows['post'][0], monotonic_ns=300,
                                     owned_registry_complete=True, owned_processes_remaining=[]))
        lease = dict(node, step=self.rows['gpu'][0], monotonic_ns=200, holder_inventory_basis='kernel_KFD_proc_only',
                     pci_to_kfd={pci: rank+1 for rank, pci in enumerate(self.pci)})
        self.write('worker-assigned.json', dict(request=reader.REQUEST, epoch=self.epoch, actual_step=self.rows['gpu'][0],
             boot_id=node['boot_id'], visible_PCI_by_index=self.pci, leased_snapshot=lease, actual_CPU_affinity=list(range(64))))
        assets = (reader.CORE / 'ASSETS.json').read_bytes()
        self.write('src/ASSETS.json', assets)
        initial = json.loads(assets)['initial']
        arms = []
        for arm, steps in reader.CHECKPOINTS.items():
            for step in steps:
                prefix = f'{arm}/train/checkpoint-{step}'
                pins = {}
                for name in reader.FILES:
                    value = b'fixture-checkpoint-body'
                    if name == 'trainer_state.json':
                        value = dict(global_step=step, max_steps=25, epoch=step/25)
                    if name == 'args.json':
                        value = dict(max_steps=25, gradient_accumulation_steps=4, per_device_train_batch_size=1)
                    pins[name] = self.write(f'{prefix}/{name}', value)
                self.write(f'{arm}-checkpoint-{step}-files.json', dict(arm=arm, global_step=step, member_count=14,
                           actual_tensor_deserialization_here=False, files=pins))
                if step == 20:
                    self.write('resumed-witness-config.json', dict(checkpoint=str(self.root / prefix), checkpoint_files=pins))
            start = 0 if arm == 'continuous' else 20
            self.write(f'{arm}/native-intent.json', dict(request=reader.REQUEST, arm=arm, epoch=self.epoch,
                                                        deadline=self.epoch+1980))
            ranks = []
            for rank in range(8):
                folder = f'{arm}/train/minicpm-usgs-witness/rank-{rank:02d}'
                inputs = [dict(ordinal=i+1, global_step=start+i//4, full_input=str(4*start+i)) for i in range(100-4*start)]
                outputs = [dict(ordinal=row['ordinal'], loss='loss', loss_scalar=.5) for row in inputs]
                updates = [dict(global_step=i, state={'optimizer': i}, optimizer_step_min=i, optimizer_step_max=i,
                                optimizer_slot_count=1, adapter_changed=i>1) for i in range(start+1,26)]
                frozen = {'model.vision_tower': {'tensors': 1, 'parameters': 2},
                          'model.merger': {'tensors': 1, 'parameters': 3}}
                result = dict(rank=rank, world_size=8, completed_global_step=25, saved_checkpoints=list(steps),
                              inputs=inputs, outputs=outputs, steps=updates, frozen=frozen)
                if arm == 'continuous':
                    self.write(f'{folder}/initialization.json', dict(named_parameters_SHA256=initial['named_SHA256'],
                        file_SHA256=initial['SHA256'], file_bytes=initial['bytes'], parameter_numel=10822656,
                        parameter_tensors=1, global_step=0, optimizer_state_entries=0))
                    self.write(f'{folder}/checkpoint-20-rng.json', dict(rank=rank, RNG='fixture-checkpoint-20'))
                else:
                    result.update(native_resume_window_complete=True, RNG_loads=[dict(saved_equal=True, global_step=20,
                                                                                    actual='fixture-checkpoint-20')])
                    self.write(f'{folder}/native-data-skip.json', dict(rank=rank, before=100, skipped=80, after=20))
                    self.write(f'{folder}/restored-before-RNG.json', dict(restored_parameter_optimizer_scheduler_equal=True,
                                                                          actual={'optimizer': 20}))
                    self.write(f'{folder}/frozen-boundaries.json', frozen)
                self.write(f'{folder}/checkpoint-25-rng.json', dict(rank=rank, RNG='fixture-checkpoint-25'))
                name = f'{folder}/result.json'
                ranks.append(dict(rank=rank, complete=True, result_file=dict(path=str(self.root / name), **self.write(name, result))))
            outcome = dict(request=reader.REQUEST, arm=arm, passed=True, cleanup_known=True, rc=0,
                           cleanup=dict(owned_processes_remaining=0, termination_needed=False), ranks=ranks)
            self.write(f'{arm}/native-outcome.json', outcome)
            arms.append(outcome)
        self.write('worker-outcome.json', dict(request=reader.REQUEST, epoch=self.epoch, passed=True,
                   native_deadline=self.epoch+1980, arms_not_attempted=[], arms=arms))
        spec = importlib.util.spec_from_file_location('fixture_compare', reader.CORE / 'compare_recovery.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.write('paired-recovery.json', module.compare(self.root))

    def test_full_fixture_rebuilds_fixed_denominators_and_numerical_tail(self):
        self.materialize()
        result = self.audit()
        self.assertTrue(result['passed'])
        self.assertEqual(result['optimizer_updates_verified'], 30)
        self.assertEqual(result['paired_rank_microbatches'], 160)
        self.assertEqual(len(result['comparison']['evidence_SHA256']), 56)
        self.assertEqual(result['checkpoint_tensor_payload_read_bytes'], 0)
        self.assertFalse(result['formal17_qualified'])

    def test_missing_live_row_and_reported_failure_keep_denominator(self):
        self.rows.pop('post')
        self.rows['gpu'][2] = 'RUNNING'
        self.assertEqual(self.audit()['state'], 'ORIGINAL_ACCOUNTING_INCOMPLETE')
        self.rows['gpu'][2] = 'FAILED'
        result = self.audit()
        self.assertEqual(result['state'], 'ORIGINAL_SLURM_ACCOUNTING_FAILED')
        self.assertEqual(result['planned_updates'], 30)
        self.assertIsNone(result['optimizer_updates_verified'])

    def test_resource_contract_is_taken_from_independent_accounting(self):
        self.rows['gpu'][6] = self.rows['gpu'][6].replace('gres/dcu=8', 'gres/dcu=4')
        self.assertEqual(self.audit()['state'], 'ORIGINAL_SLURM_ACCOUNTING_FAILED')

    def test_bad_cursor_and_changed_checkpoint_are_rejected(self):
        self.materialize()
        wrong = 'resumed/train/minicpm-usgs-witness/rank-03/native-data-skip.json'
        original = (self.root / wrong).read_bytes()
        self.write(wrong, dict(rank=3, before=100, skipped=79, after=21))
        with self.assertRaisesRegex(ValueError, 'actual native skip'):
            self.audit()
        self.write(wrong, original)
        self.write('continuous/train/checkpoint-20/rng_state_7.pth', b'wrong checkpoint size')
        with self.assertRaisesRegex(ValueError, 'checkpoint member'):
            self.audit()


if __name__ == '__main__':
    unittest.main()
