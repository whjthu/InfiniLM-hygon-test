from copy import deepcopy
import json
import unittest

from read_deep_c4_direct_history_link_terminal import TASK, accounting, bind_host, read_completed
from verify_deep_c4_direct_history_link import (CANDIDATE_NATIVE, CANDIDATE_RECEIPT_SHA, PLAN, PLAN_SHA,
    PROBE_REPORT, PROBE_REPORT_SHA, REPO, link_argv, read_pinned_json)


class DirectHistoryTerminalTest(unittest.TestCase):
    def setUp(self):
        self.claim = json.loads((REPO / TASK).read_bytes())
        self.plan = read_pinned_json(REPO / PLAN, PLAN_SHA)
        self.original = read_pinned_json(REPO / PROBE_REPORT, PROBE_REPORT_SHA)['independent_original_build']['result']
        link, = self.plan['links']
        base = link['required_NEEDED']
        self.upstream_sha, self.artifact_sha = 'c' * 64, 'd' * 64
        self.needed = [CANDIDATE_NATIVE if name == base else name for name in self.original['needed']]
        self.placement = self.claim['placement']
        pub = 'a' * 40
        self.intent = dict(request=self.claim['request'], publication=pub, manifest_SHA256=self.claim['manifest_SHA256'],
                           parent=self.placement['parent'], node=self.placement['node'], GPU=False,
                           epoch=2000.0, deadline=2300.0, replay_allowed=False)
        self.launch = dict(self.intent, pid=123, pgid=123, startticks=50, host='owner-host-fixture')
        self.binding = bind_host(self.claim, self.intent, self.launch, pub)
        step = dict(self.binding, step='1103606.901', new_translation_units=0, actual_GPU_calls=0,
                    checkpoint_weight_reads=0, runtime_provider_verified=False, GPU_GO=False, Model_GO=False)
        self.files = {
            'CPU-result.json': dict(step, completed=True, passed=True, linker_attempts=1, linker_returncode=0,
                                    guarded_probe_object_reused=True, relinked_probe_SHA256=self.artifact_sha,
                                    expected_DT_NEEDED=self.needed),
            'intent.json': dict(step, replay_allowed=False, guarded_probe_object_reuse_planned=True),
            'link-plan.json': dict(linker_argv=link_argv(self.plan, link),
                                   candidate_receipt_SHA256=CANDIDATE_RECEIPT_SHA,
                                   expected_DT_NEEDED=self.needed, input_SHA256={
                                       base: self.original['project_provider_SHA256'][base],
                                       self.plan['candidate'] + '/probe.o': self.original['object_SHA256'],
                                       self.plan['candidate'] + '/' + link['output']: self.original['probe_SHA256'],
                                       CANDIDATE_NATIVE: self.upstream_sha,
                                   }),
            'link-attempt.json': dict(request=self.binding['request'], replay_allowed=False,
                                      original_deadline=self.binding['deadline'], linker_timeout_seconds=80),
        }

    def row(self, state='COMPLETED', code='0:0', tres='cpu=4,mem=8G,node=1'):
        return f"1103606.901|{self.claim['label']}|{state}|{code}|12|f13r2n01|{tres}\n"

    def consume(self, rows, read=None, artifact=None):
        def fixture(name):
            if name not in self.files:
                raise FileNotFoundError(name)
            return deepcopy(self.files[name]), 'f' * 64
        return read_completed(accounting(rows, self.claim), self.binding, read or fixture,
                              artifact or (lambda: self.artifact_sha), lambda: self.upstream_sha,
                              self.original, self.plan)

    def test_completed_exact_step_and_original_one_link_artifact_are_bound(self):
        result = self.consume(self.row())
        self.assertEqual(result['state'], 'ORIGINAL_CPU_LINK_COMPLETED_ARTIFACT_MATCHED')
        self.assertTrue(result['cpu_link_artifact_qualified'])
        self.assertEqual(result['probe_SHA256'], self.artifact_sha)
        self.assertEqual(result['candidate_native_SHA256'], self.upstream_sha)
        self.assertFalse(result['GPU_GO'])
        self.assertFalse(result['runtime_provider_verified'])

    def test_running_or_missing_accounting_never_reads_output(self):
        for rows in (self.row('RUNNING'), ''):
            result = self.consume(rows, read=lambda _: self.fail('must not open output'))
            self.assertFalse(result['output_consumed'])
            self.assertEqual(result['state'], 'ORIGINAL_TERMINAL_NOT_OBSERVED')

    def test_failed_accounting_stays_failed_despite_successful_link_receipt(self):
        result = self.consume(self.row('TIMEOUT', '0:15'), artifact=lambda: self.fail('must not inspect output'))
        self.assertFalse(result['cpu_link_artifact_qualified'])
        self.assertTrue(result['output_consumed'])
        del self.files['CPU-result.json']
        self.assertEqual(self.consume(self.row('CANCELLED by 6', '0:15'))['state'],
                         'ORIGINAL_SLURM_FAILED_RESULT_MISSING')
        with self.assertRaisesRegex(ValueError, 'completed CPU step has no original result'):
            self.consume(self.row())

    def test_duplicate_wrong_resources_and_wrong_process_identity_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'repeated exact'):
            self.consume(self.row() * 2)
        with self.assertRaisesRegex(ValueError, 'zero-GPU'):
            self.consume(self.row(tres='cpu=4,mem=8G,node=1,gres/dcu=1'))
        with self.assertRaisesRegex(ValueError, 'owned launcher'):
            bind_host(self.claim, self.intent, dict(self.launch, pgid=999), self.binding['publication'])

    def test_child_attempt_upstream_and_digest_drift_cannot_qualify(self):
        with self.assertRaisesRegex(ValueError, 'relinked probe bytes differ'):
            self.consume(self.row(), artifact=lambda: 'e' * 64)
        self.files['link-plan.json']['input_SHA256'][CANDIDATE_NATIVE] = 'e' * 64
        with self.assertRaisesRegex(ValueError, 'original object or linkage pins'):
            self.consume(self.row())
        self.files['link-plan.json']['input_SHA256'][CANDIDATE_NATIVE] = self.upstream_sha
        self.files['link-attempt.json']['original_deadline'] += 1
        with self.assertRaisesRegex(ValueError, 'bounded link attempt differs'):
            self.consume(self.row())


if __name__ == '__main__':
    unittest.main()
