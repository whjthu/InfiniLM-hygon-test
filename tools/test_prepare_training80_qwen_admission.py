from copy import deepcopy
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import prepare_training80_qwen_admission as adapter
from preflight_qwen_usgs_exposure25 import preflight
from record_qwen_usgs_dev80_absence import read_record
from reconcile_qwen_usgs_dev80_original import AUTH_RECEIPT, CLAIM as DEV80_CLAIM
from report_training80_backlog import REPO, invocation, verify_launch
from test_preflight_qwen_usgs_exposure25 import LIVE


class QwenAdmissionTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.repo = Path(temp.name).resolve()
        self.private = self.repo/'local-only'
        self.private.mkdir()
        self.target = self.repo/'target'
        self.target.mkdir()
        candidate = dict(id=adapter.CANDIDATE, case=adapter.CASE, kind='training', state='blocked',
                         priority=14, placement={adapter.NODE: 8}, min_updates=30,
                         attempt_receipts=adapter.MARKERS, next_action='fixture pending admission')
        self.backlog = dict(minimum_ready_training=5, items=[candidate])
        for path in (adapter.SOURCE, AUTH_RECEIPT, DEV80_CLAIM):
            target = self.repo/path
            target.parent.mkdir(parents=True, exist_ok=True)
            if (REPO/path).is_dir():
                shutil.copytree(REPO/path,target)
            else:
                shutil.copyfile(REPO/path,target)
        self.original_blobs = {name: adapter.original_blob(REPO,name) for name in ('CLAIM.json','SOURCE_FILES.json')}
        for name, raw in self.original_blobs.items():
            (self.repo/adapter.SOURCE/name).write_bytes(raw)
        self.read_original = lambda _,name: self.original_blobs[name]
        original = json.loads(self.original_blobs['CLAIM.json'])
        for item in original['public_prerequisites']:
            target = self.repo/item['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO/item['path'],target)
        self.readout = preflight(self.repo, self.target, account=lambda: LIVE,
            membership=lambda _: dict(observed_epoch=1000.,
                                      node_membership=dict.fromkeys(adapter.DEV_NODES.values(),'NO_NUMBERED_STEP')),
            verify=lambda *_: dict(framework_files=393, train_rows=88, train_media_files=22, model_shard_metadata=4,
                                  small_model_files=10, CPU_input_log_bytes=867, model_weight_body_bytes_read=0,
                                  initial_state_metadata=True, initial_state_body_bytes_read=0))
        self.readout['returned_epoch'] = 1001.

    def apply_adoption(self):
        result = adapter.prepare_adoption(self.repo, adapter.as_json(self.readout), self.private/'adoption',
                                          read_original=self.read_original)
        for key,name in (('claim','CLAIM.json'),('manifest','SOURCE_FILES.json')):
            shutil.copyfile(self.repo/result[key]['path'], self.repo/adapter.SOURCE/name)

    def overlay(self, readout=None):
        def publication(repo, commit, paths):
            self.assertEqual(repo, self.repo)
            self.assertEqual(commit, '7'*40)
            self.assertEqual(len(paths), 23)
            self.assertIn((adapter.SOURCE/'launch.py').as_posix(), paths)
        return adapter.prepare(self.repo, self.backlog, adapter.as_json(readout or self.readout),
                               self.private/'overlay', '7'*40, check_publication=publication,
                               read_original=self.read_original)

    def test_exact_original_claim_changes_only_review_fields_and_enables_frozen_queue_invocation(self):
        self.apply_adoption()
        claim = json.loads((self.repo/adapter.SOURCE/'CLAIM.json').read_bytes())
        original = json.loads(self.original_blobs['CLAIM.json'])
        self.assertEqual({k for k in original if original[k] != claim[k]},
                         {'GPU_GO','root_GO','training_GO','state'})
        result = self.overlay()
        overlay = json.loads((self.repo/result['backlog']['path']).read_bytes())
        [job] = [job for job in overlay['items'] if job['id'] == adapter.CANDIDATE]
        self.assertEqual(job['state'], 'ready')
        self.assertEqual(job['launch']['required_ports'], [29918])
        self.assertEqual(verify_launch(self.repo,job), 'ready')
        self.assertEqual(invocation(self.repo,job)['arguments'], ['7'*40])
        self.assertIsNotNone(read_record(self.repo))
        self.assertEqual(result['actual_slurm_submissions'], 0)
        self.assertFalse(any(x.get('state') == 'ready' for x in self.backlog['items']))

    def test_late_original_handle_wrong_scope_or_unverified_assets_cannot_adopt(self):
        for field,key,new in (('last','current_rows',[['1103606.512']]), ('first','clear',False),
                              ('assets','train_rows',87), ('assets','model_weight_body_bytes_read',1),
                              ('membership','observed_epoch',1100.)):
            with self.subTest(key=key):
                value = deepcopy(self.readout)
                value[field][key] = new
                with self.assertRaises(ValueError):
                    adapter.prepare_adoption(self.repo, adapter.as_json(value), self.private/'bad',
                                             read_original=self.read_original)
                self.assertFalse((self.private/'bad').exists())

    def test_departed_source_or_recorded_attempt_prevents_overlay(self):
        self.apply_adoption()
        worker = self.repo/adapter.SOURCE/'launch.py'
        original = worker.read_bytes()
        worker.write_bytes(original+b'\n')
        with self.assertRaisesRegex(ValueError, 'adopted input differs'):
            self.overlay()
        worker.write_bytes(original)
        intent = self.repo/adapter.MARKERS[-1]
        intent.parent.mkdir()
        intent.write_text('{}',encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'already attempted'):
            self.overlay()
        self.assertFalse((self.private/'overlay').exists())
        self.assertIsNone(read_record(self.repo))


if __name__ == '__main__':
    unittest.main()
