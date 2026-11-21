from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import read_vggt_tum_bidirectional30_terminal as reader


class TerminalReaderTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.epoch = 1795220000.0
        self.claim, self.manifest, self.publication = 'a'*64, 'b'*64, 'c'*40
        self.rows = {phase: [f'1103606.{i}', reader.LABEL+'-'+phase, 'COMPLETED', '0:0', '20',
                             reader.NODE, tres] for i, (phase, tres) in enumerate((
                     ('pre','cpu=2,mem=2G,node=1'), ('gpu','cpu=4,mem=64G,node=1,gres/dcu=1'),
                     ('post','cpu=2,mem=2G,node=1')), 610)}
        self.boot = 'fixture-original-boot'
        self.put('intent.json', dict(request=reader.REQUEST, parent=reader.PARENT, node=reader.NODE,
                 epoch=self.epoch, deadline=self.epoch+1800, source_manifest_SHA256=self.manifest,
                 claim_SHA256=self.claim, publication=self.publication))
        pci = [f'0000:0{i}:00.0' for i in range(8)]
        for tick, (phase, name) in enumerate((('pre','before.json'), ('post','after.json')), 1):
            self.put(name, dict(step=self.rows[phase][0], node=reader.NODE, mode='host_complete', pci=pci,
                               boot_id=self.boot, monotonic_ns=tick, full_fd_inventory_performed=True,
                               kfd_holder_count=0))
        self.put('assignment.json', dict(step=self.rows['gpu'][0], node=reader.NODE, pci=pci[0], pid=800,
                 startticks=709, kfd=dict(pci=pci[0], gpu_id=111, properties_SHA256='e'*64)))

    def put(self, name, value):
        raw = value if isinstance(value, bytes) else json.dumps(value).encode()
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return sha256(raw).hexdigest()

    def arms(self, loss_error=0):
        receipts, resume = {}, None
        for ordinal, (arm, first) in enumerate(reader.ARMS.items()):
            base = f'arms/{arm}/'
            identity = dict(arm=arm, epoch=self.epoch, deadline=self.epoch+1700,
                            process=dict(boot_id=self.boot, pid=810+ordinal, startticks=720+ordinal))
            self.put(base+'invocation.json', identity)
            result = {'metrics': dict(global_optimizer_step=25, executed_optimizer_steps=25-first,
                completed_global_step_before_resume=first, cached_valid_batches=4, sequence_count=4,
                measurement_steps=20-max(0,first-5), formal_CO3D_eligible=False, resume_source_sha256=resume,
                checkpoint_identity_evidence=dict(frozen_model=dict(weights_model_pt=reader.WEIGHT)),
                relative_translation_loss=reader.DIRECTIONAL, checkpoint_identity='target-contract',
                final_state_sha256='d'*64), 'checks': dict.fromkeys((
                'camera_head_weights_changed','frozen_aggregator_unchanged','main_forward_device',
                'nonzero_finite_camera_head_gradients','resume_boundary'), True)}
            result_sha = self.put(base+'native-result.json', result)
            rows = [dict(global_optimizer_step=step, batch_cursor=(step-1)%4, loss=.1+ordinal*loss_error,
                         grad_norm=.2, camera_head_grad_norm_before_clip=.3, relative_translation_direction_cost=.4)
                    for step in range(first+1, 26)]
            self.put(base+'train/steps.ndjson', b''.join(json.dumps(row).encode()+b'\n' for row in rows))
            checkpoints = {}
            for n in (10, 20, 25):
                if n <= first:
                    continue
                checkpoint_hash = self.put(base+f'train/checkpoint-{n}.pt', f'{arm}-{n}'.encode())
                checkpoints[str(n)] = dict(global_step=n, next_global_step=n+1, next_batch_cursor=n%4,
                    file_SHA256=checkpoint_hash, optimizer_group_learning_rates=[2e-5],
                    **dict.fromkeys(reader.COMPONENTS,'f'*64))
            receipt = dict(identity, bundle_SHA256=reader.BUNDLE, backend_SHA256=reader.BACKEND,
                          gradient_accumulation=1, executed_optimizer_steps=25-first, physical_training_pairs=2,
                          physical_scene_count=2, ordered_training_views=4, formal_CO3D_eligible=False,
                          scheduler=dict(type='constant_learning_rate_in_optimizer_param_groups',
                                         mutable_scheduler=False,learning_rate=2e-5), result_SHA256=result_sha,
                          resume_source_SHA256=resume, final_state_SHA256='d'*64, checkpoint_identity='target-contract',
                          steps=rows, checkpoints=checkpoints,
                          ordered_view_exposures=[sum(row['batch_cursor']==i for row in rows) for i in range(4)])
            if not first:
                resume = checkpoints['20']['file_SHA256']
            receipts[arm] = dict(SHA256=self.put(base+'receipt.json',receipt), value=receipt)
        native = self.put('native-result.json', {'receipts':receipts})
        self.put('outcome.json', dict(request=reader.REQUEST, publication=self.publication, formal_acceptance=False,
                 effect_qualified=False, passed=True, result_SHA256=native,
                 stages={phase:dict(passed=True,row=row) for phase,row in self.rows.items()}))

    def run_reader(self):
        raw = '\n'.join('|'.join(row) for row in self.rows.values()).encode()
        return reader.verify(self.root,raw,self.manifest,self.claim)

    def test_reads_original_denominator_end_to_end(self):
        self.arms()
        result=self.run_reader()
        self.assertTrue(result['passed'])
        self.assertEqual(result['independently_verified_updates'],30)
        self.assertEqual(len([name for name in result['original_file_SHA256'] if name.endswith('.pt')]),4)
        self.assertFalse(result['formal_CO3D_eligible'])

    def test_accounting_running_cannot_claim_cleanup(self):
        self.rows['gpu'][2]='RUNNING'
        result=self.run_reader()
        self.assertFalse(result['gpu_terminal'])
        self.assertFalse(result['independent_post_clean'])
        self.assertEqual(result['independently_verified_updates'],0)

    def test_bad_resume_checkpoint_retains_original_25_updates(self):
        self.arms()
        self.put('arms/resume20/train/checkpoint-25.pt',b'foreign checkpoint')
        result=self.run_reader()
        self.assertFalse(result['passed'])
        self.assertTrue(result['independent_post_clean'])
        self.assertEqual(result['independently_verified_updates'],25)
        self.assertIn('checkpoint content',result['errors']['resume20'])

    def test_both_completed_denominators_do_not_hide_recovery_drift(self):
        self.arms(loss_error=.001)
        result=self.run_reader()
        self.assertEqual(result['independently_verified_updates'],30)
        self.assertTrue(result['independent_post_clean'])
        self.assertFalse(result['recovery']['passed'])
        self.assertFalse(result['passed'])

    def test_clean_post_survives_original_gpu_failure(self):
        self.rows['gpu'][2:4]=['FAILED','1:0']
        result=self.run_reader()
        self.assertTrue(result['gpu_terminal'])
        self.assertTrue(result['independent_post_clean'])
        self.assertFalse(result['passed'])

    def test_inconsistent_collector_preserves_reported_pass_and_independent_evidence(self):
        self.arms()
        outcome=json.loads((self.root/'outcome.json').read_bytes())
        outcome['result_SHA256']='0'*64
        self.put('outcome.json',outcome)
        result=self.run_reader()
        self.assertTrue(result['original_collector_reported_passed'])
        self.assertFalse(result['original_collector_verified'])
        self.assertEqual(result['independently_verified_updates'],30)
        self.assertTrue(result['independent_post_clean'])
        self.assertFalse(result['passed'])


if __name__ == '__main__':
    unittest.main()
