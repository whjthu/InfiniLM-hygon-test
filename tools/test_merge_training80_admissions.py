from copy import deepcopy
import json
import unittest

from merge_training80_admissions import RECEIPTS, merge
from prepare_training80_dispatch_handoff import build_handoff
from prepare_training80_dispatch_probe_args import probe_arguments
import test_report_training80_backlog
from tuning_queue import TRAINING80_LEDGER


class AdmissionMergeTest(unittest.TestCase):
    def setUp(self):
        self.fixture = test_report_training80_backlog.BacklogTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.repo = self.fixture.repo
        self.jobs = [self.admission(1, 'a01r4n16'), self.admission(2, 'f13r4n09'),
                     self.admission(3, 'f13r1n19')]
        items = []
        for job in self.jobs:
            blocked = dict(job, state='blocked', attempt_receipts=job['launch']['attempt_receipts'])
            del blocked['launch']
            items.append(blocked)
        self.base = dict(minimum_ready_training=5, items=items)
        self.paths = [self.overlay(job) for job in self.jobs]

    def admission(self, index, node):
        job = self.fixture.trial(index, node=node)
        preflight = self.fixture.put(f'local-only/preflight-{index}.json', dict(request=job['launch']['request']))
        for key in RECEIPTS:
            ref = job['launch'][key]
            body = json.loads((self.repo / ref['path']).read_bytes())
            job['launch'][key] = self.fixture.put(ref['path'], dict(body, target_preflight=preflight))
        return dict(job, attempt_receipts=job['launch']['attempt_receipts'])

    def overlay(self, *jobs, name=None):
        replacements = {job['id']: job for job in jobs}
        if name is None:
            name = '-'.join(replacements)
        items = [replacements.get(row['id'], row) for row in self.base['items']]
        return self.fixture.put(f'local-only/overlay-{name}.json', dict(self.base, items=items))['path']

    def test_independent_target_admissions_feed_one_probe_and_one_handoff(self):
        result, report = merge(self.repo, self.base, self.paths[:2])
        self.assertEqual(report['accepted'], {'trial-1': 'ready', 'trial-2': 'ready'})
        self.assertEqual(report['rejected'], {})
        self.assertEqual(result['items'][2], self.base['items'][2])
        arguments = probe_arguments(self.repo, result, json.loads(TRAINING80_LEDGER.read_bytes()), ())
        self.assertEqual(arguments['verified_training_ready'], 2)
        self.assertEqual({node for nodes in arguments['priority_by_parent'].values() for node in nodes},
                         {'a01r4n16', 'f13r4n09'})
        plan = build_handoff(self.fixture.verified_snapshot(), result, repo=self.repo,
                             now=self.fixture.now, boot_anchor=self.fixture.anchor)
        self.assertEqual(plan['training_ready_deficit'], 3)
        self.assertEqual([packet['candidate'] for packet in plan['coordinator_handoffs']], ['trial-1', 'trial-2'])
        self.assertEqual(plan['actual_dispatched_DCU'], 0)

    def test_prior_intent_stays_held_and_is_excluded_from_ready_capacity(self):
        self.fixture.put(self.jobs[0]['launch']['attempt_receipts'][0], dict(remote_outcome='UNKNOWN'))
        result, report = merge(self.repo, self.base, self.paths[:2])
        self.assertEqual(report['accepted'], {'trial-1': 'reconcile', 'trial-2': 'ready'})
        plan = build_handoff(self.fixture.verified_snapshot(), result, repo=self.repo,
                             now=self.fixture.now, boot_anchor=self.fixture.anchor)
        self.assertEqual(plan['outstanding_training_ready'], 1)
        self.assertEqual(plan['original_launch_holds']['trial-1']['nodes'], ['a01r4n16'])
        self.assertEqual([packet['candidate'] for packet in plan['coordinator_handoffs']], ['trial-2'])

    def test_changed_scope_and_missing_original_preflight_do_not_block_other_training(self):
        corrupt = deepcopy(self.jobs[2])
        corrupt['min_updates'] += 1
        bad_scope = self.overlay(corrupt, name='changed-scope')
        self.fixture.put('local-only/preflight-1.json', dict(request='different'))
        result, report = merge(self.repo, self.base, [self.paths[0], bad_scope, self.paths[1]])
        self.assertEqual(report['accepted'], {'trial-2': 'ready'})
        self.assertIn('evidence SHA256 differs', report['rejected']['trial-1'])
        self.assertIn('frozen candidate scope', report['rejected'][bad_scope])
        self.assertEqual(result['items'][0], self.base['items'][0])

    def test_duplicate_workloads_and_conflicts_keep_independent_candidate(self):
        duplicate = deepcopy(self.jobs[2])
        duplicate['launch']['workload_SHA256'] = self.jobs[0]['launch']['workload_SHA256']
        key = 'prerequisite_receipt'
        ref = duplicate['launch'][key]
        body = json.loads((self.repo / ref['path']).read_bytes())
        body['workload_SHA256'] = duplicate['launch']['workload_SHA256']
        duplicate['launch'][key] = self.fixture.put(ref['path'], body)
        path = self.overlay(duplicate, name='duplicate-workload')
        _, report = merge(self.repo, self.base, [*self.paths[:2], path])
        self.assertEqual(report['accepted'], {'trial-2': 'ready'})
        self.assertEqual(set(report['rejected']), {'trial-1', 'trial-3'})
        conflict = self.overlay(dict(self.jobs[0], next_action='different local plan'), name='conflict')
        _, report = merge(self.repo, self.base, [*self.paths[:2], conflict])
        self.assertEqual(report['accepted'], {'trial-2': 'ready'})
        self.assertIn('conflicting local admissions', report['rejected']['trial-1'])


if __name__ == '__main__':
    unittest.main()
