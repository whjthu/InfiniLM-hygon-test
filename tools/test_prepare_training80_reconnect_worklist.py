from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from prepare_training80_reconnect_worklist import prepare
from tuning_queue import TRAINING80_LEDGER


class ReconnectWorklistTest(unittest.TestCase):
    def setUp(self):
        working = tempfile.TemporaryDirectory()
        self.addCleanup(working.cleanup)
        self.repo = Path(working.name)
        self.ledger = TRAINING80_LEDGER.read_bytes()

    def write(self, filename, content):
        payload = content if isinstance(content, bytes) else json.dumps(content).encode()
        path = self.repo / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return dict(path=filename, SHA256=sha256(payload).hexdigest(), bytes=len(payload))

    def candidate(self, suffix='one', *, request='a' * 32, cards=8, port=29918, parent='1103606'):
        source = f'code/bw1000/scope-{suffix}'
        claim = dict(request=request, node='f13r1n10', parent=parent, device_count=cards, cpus=4,
                     GPU_GO=False, CPU_GO=cards == 0, optimizer_updates=25 if cards else 0)
        if port is not None:
            claim['master_port'] = port
        claim_ref = self.write(source + '/CLAIM.json', claim)
        launcher = self.write(source + '/launch.py', b'def main(publication, transport): pass\n')
        member_pins = {Path(ref['path']).name: {key: value for key, value in ref.items() if key != 'path'}
                       for ref in (claim_ref, launcher)}
        manifest = self.write(source + '/SOURCE_FILES.json', dict(files=member_pins))
        task = self.write(f'coordination/tasks/scope-{suffix}.json', dict(state='SOURCE_PENDING',
            case='T01.domain', request=request, claim=claim_ref, manifest=manifest))
        return dict(task=task, expected_task_state='SOURCE_PENDING', source=source, launcher=launcher,
                    attempt_receipt=f'local-only/scope-{suffix}/intent.json', gates=['check target'])

    def plan(self, *candidates):
        return prepare(self.repo, dict(candidates=list(candidates)), self.ledger)

    def test_source_preparation_does_not_invent_live_capacity(self):
        training = self.candidate()
        cpu = self.candidate('cpu', request='b' * 32, cards=0, port=None)
        result = self.plan(training, cpu)
        self.assertEqual(result['collector_port_arguments'], ['--port', '29918'])
        self.assertEqual(result['candidate_nodes_needing_current_probe'], ['f13r1n10'])
        self.assertEqual(result['configured_DCU'], 640)
        self.assertIsNone(result['current_allocation_DCU'])
        self.assertIsNone(result['current_available_DCU'])
        self.assertIsNone(result['measured_busy_percent'])
        self.assertEqual(result['newly_verified_training_ready'], 0)
        self.assertEqual(result['candidates'][1]['planned_optimizer_updates'], 0)

    def test_original_attempt_moves_to_reconciliation_and_omits_new_probe(self):
        training = self.candidate()
        self.write(training['attempt_receipt'], dict(transport_rc=255))
        report = self.plan(training)
        self.assertEqual(report['collector_port_arguments'], [])
        self.assertEqual(report['candidate_nodes_needing_current_probe'], [])
        self.assertEqual(report['candidates'][0]['target_preflight_state'], 'RECONCILE_ORIGINAL_ATTEMPT')

    def test_manifest_pin_rejects_one_changed_launcher(self):
        training = self.candidate()
        self.write(training['launcher']['path'], b'def main(external_process): pass\n')
        with self.assertRaisesRegex(ValueError, 'source member bytes differ'):
            self.plan(training)

    def test_read_only_preflight_requires_pinned_dependencies_and_no_original_attempt(self):
        training = self.candidate()
        entry = self.write('tools/exact_preflight.py', b'print("readonly")\n')
        dependency = self.write('coordination/training/ledger.json', {'nodes': 80})
        probe = dict(entry=entry, dependencies=[dependency])
        training['target_read_only_preflight'] = probe
        self.assertEqual(self.plan(training)['candidates'][0]['target_read_only_preflight'], probe)
        self.write(training['attempt_receipt'], dict(transport_rc=255))
        self.assertIsNone(self.plan(training)['candidates'][0]['target_read_only_preflight'])
        self.write(dependency['path'], {'nodes': 40})
        with self.assertRaisesRegex(ValueError, 'evidence SHA256 differs'):
            self.plan(training)

    def test_prior_request_cannot_authorize_same_request_or_replay(self):
        training = self.candidate()
        prior = self.write('results/bw1000/old.json', dict(request='b' * 32, replay_allowed=False))
        training['blocking_original_requests'] = [dict(prior, request='b' * 32)]
        [report] = self.plan(training)['candidates']
        self.assertFalse(report['blocking_original_requests'][0]['replay_allowed'])
        changed = deepcopy(training)
        changed['blocking_original_requests'][0]['request'] = 'a' * 32
        with self.assertRaisesRegex(ValueError, 'original request reference'):
            self.plan(changed)

    def test_rejects_wrong_parent_and_deduplicates_requests(self):
        with self.assertRaisesRegex(ValueError, 'assigned Slurm parent'):
            self.plan(self.candidate(parent='1087656'))
        with self.assertRaisesRegex(ValueError, 'repeated or mismatched'):
            self.plan(self.candidate(), self.candidate('duplicate'))


if __name__ == '__main__':
    unittest.main()
