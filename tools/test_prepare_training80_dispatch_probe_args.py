import json
import subprocess
import unittest
from unittest.mock import patch

from collect_training80_dispatch_snapshot import collect, probe_batches
from prepare_training80_dispatch_probe_args import probe_arguments
from report_training80_backlog import TRAINING80_LEDGER
import test_report_training80_backlog as backlog_fixtures
import test_training_slurm_membership as membership_fixtures


class DispatchProbeArgumentsTest(unittest.TestCase):
    def setUp(self):
        self.fixture = backlog_fixtures.BacklogTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.ledger = json.loads(TRAINING80_LEDGER.read_bytes())

    def test_training_precedes_operator_and_source_only_does_not_count(self):
        operator = self.fixture.trial(1, 'operator', node='f13r1n11')
        training = self.fixture.trial(2, node='f13r1n10')
        preflight = dict(case='T12.sft', node='f13r4n09', requested_DCU=1,
                         target_preflight_state='WAITING_AUTHENTICATED_TARGET', frozen_rendezvous_port=29918)
        report = probe_arguments(self.fixture.repo, {'items': [operator, training]}, self.ledger, [], [preflight],
                                 priority_only=True)
        self.assertEqual(report['priority_by_parent']['1103606'], ['f13r1n10', 'f13r1n11', 'f13r4n09'])
        self.assertEqual(report['required_ports'], [29907, 29918])
        self.assertEqual(report['verified_training_ready'], 1)
        self.assertEqual(report['training_ready_without_original_hold'], 1)
        self.assertEqual(report['prepared_preflight_priority_nodes'], ['f13r4n09'])
        self.assertEqual(report['collector_arguments'][-1], '--priority-only')
        self.assertIsNone(report['current_device_eligible_DCU'])

    def test_one_held_node_excludes_whole_multi_node_placement_and_port(self):
        training = self.fixture.trial(1, node='f13r1n10')
        training['placement']['f13r1n11'] = 8
        launch = training['launch']
        launch['prerequisite_receipt'] = self.fixture.put(launch['prerequisite_receipt']['path'], dict(
            request=launch['request'], state='TARGET_PREREQUISITES_VERIFIED', workload_SHA256=launch['workload_SHA256'],
            min_updates=training['min_updates'], placement=training['placement'], required_ports=[29907]))
        report = probe_arguments(self.fixture.repo, {'items': [training]}, self.ledger, [{'nodes': ['f13r1n10']}])
        self.assertEqual(report['verified_training_ready'], 1)
        self.assertEqual(report['training_ready_without_original_hold'], 0)
        self.assertEqual(report['blocked_candidate_held_nodes'], {'trial-1': ['f13r1n10']})
        self.assertEqual(report['collector_arguments'], ['--held-node', 'f13r1n10'])

    def test_unknown_original_holds_node_even_if_candidate_source_changes(self):
        training = self.fixture.trial(1, node='f13r1n10')
        operator = self.fixture.trial(2, 'operator', node='f13r1n10')
        self.fixture.put(training['launch']['attempt_receipts'][0], dict(rc=255, remote_outcome='UNKNOWN'))
        self.fixture.put('code/bw1000/fixture-1/config.json', dict(context=1))
        report = probe_arguments(self.fixture.repo, {'items': [training, operator]}, self.ledger, [])
        self.assertEqual(report['reconcile_candidates'], {'trial-1': ['f13r1n10']})
        self.assertEqual(report['blocked_candidate_held_nodes'], {'trial-2': ['f13r1n10']})
        self.assertEqual(report['collector_arguments'], ['--held-node', 'f13r1n10'])


class DispatchProbeCollectionTest(unittest.TestCase):
    def test_priority_batches_keep_parent_ids_and_original_normal_order(self):
        nodes = {'11': ['c', 'd'], '10': ['a', 'b']}
        self.assertEqual([row[2:] for row in probe_batches(nodes, [])], [('10', 0, ['a', 'b']), ('11', 0, ['c', 'd'])])
        self.assertEqual([row[2:] for row in probe_batches(nodes, ['d', 'b'])],
                         [('10', 0, ['b']), ('11', 0, ['d']), ('10', 1, ['a']), ('11', 1, ['c'])])

    def collect_fixture(self, *, priority, held, priority_only=False):
        fixture = membership_fixtures.RealPlacementFixtureTest()
        fixture.setUp()
        for parent in fixture.groups:
            fixture.rows[parent] = [f'StepId={parent}.extern TRES=(null)']
        pins = {node: [f'0000:{bus:02x}:00.0' for bus in range(8)]
                for groups in fixture.groups.values() for nodes in groups.values() for node in nodes}
        requests = []
        views = []

        def run(args, *, capture_output, timeout, check):
            if args[0] != 'srun':
                if args[:3] == ('scontrol', 'show', 'step'):
                    views.append(args[3])
                return subprocess.CompletedProcess(args, 0, '\n'.join(fixture.query(*args)).encode(), b'')
            nodes = next(arg.split('=', 1)[1] for arg in args if arg.startswith('--nodelist=')).split(',')
            request, parent, _ = args[-3:]
            requests.append((request, nodes))
            self.assertIn('--gres=none', args)
            rows = [dict(request_id=request, parent=parent, step=parent + '.997', node=node, ports_checked=[],
                         observer_uid=1234, PCIs=pins[node], kfd_process_count=0, own_device_fd_process_count=0,
                         occupied_requested_ports=[]) for node in nodes]
            return subprocess.CompletedProcess(args, 0, '\n'.join(map(json.dumps, rows)).encode(), b'')

        request = 'a' * 32
        with patch('collect_training80_dispatch_snapshot.os.getuid', return_value=1234, create=True):
            report = collect(request, TRAINING80_LEDGER.read_bytes(), pins, [],
                             priority_nodes=priority, held_nodes=[held], priority_only=priority_only, run=run)
        self.assertEqual(sorted(views), ['1087656', '1087656', '1103606', '1103606'])
        self.assertEqual(len(set(request_id for request_id, _ in requests)), len(requests))
        return report, {node for _, nodes in requests for node in nodes}, set(pins)

    def test_held_node_stays_unobserved_during_full_fleet_collection(self):
        held = 'f13r1n10'
        priority = ['f11r4n09', 'f13r4n09']
        report, tried, nodes = self.collect_fixture(priority=priority, held=held)
        self.assertEqual(len(report['nodes']), 80)
        self.assertEqual(report['device_probe_scope'], 'FULL_FLEET')
        self.assertEqual(report['nodes'][held]['probe_error'], 'LOCAL_QUEUE_HELD')
        for key in ('device_probe_observed', 'PID_clean', 'KFD_clean'):
            self.assertFalse(report['nodes'][held][key])
        self.assertEqual(tried, nodes - {held})
        for parent in ('1087656', '1103606'):
            batches = report['probes'][parent]['batches']
            self.assertTrue(batches[0]['priority'])
            self.assertEqual(len(batches[0]['requested_nodes']), 1)
            self.assertIn(batches[0]['requested_nodes'][0], priority)
            self.assertFalse(any(batch['priority'] for batch in batches[1:]))
        self.assertIsNone(report['measured_busy_percent'])

    def test_priority_only_preserves_two_full_views_and_defers_ordinary_devices(self):
        held = 'f13r1n10'
        priority = ['f11r4n09', 'f13r4n09']
        report, tried, nodes = self.collect_fixture(priority=priority, held=held, priority_only=True)
        self.assertEqual(report['device_probe_scope'], 'PRIORITY_ONLY')
        self.assertEqual(report['membership']['distinct_nodes'], 80)
        self.assertEqual(report['membership']['allocated_DCU'], 640)
        self.assertEqual(tried, set(priority))
        self.assertEqual(report['deferred_device_probe_nodes'], sorted(nodes - tried - {held}))
        self.assertTrue(all(report['nodes'][node]['device_probe_observed'] for node in tried))
        for node in report['deferred_device_probe_nodes']:
            self.assertEqual(report['nodes'][node]['probe_error'], 'DEFERRED_BY_PRIORITY_ONLY')
            self.assertFalse(report['nodes'][node]['device_probe_observed'])
        self.assertEqual(report['nodes'][held]['probe_error'], 'LOCAL_QUEUE_HELD')
        self.assertIsNone(report['measured_busy_percent'])

    def test_priority_only_falls_back_when_only_priority_is_held(self):
        held = 'f13r1n10'
        report, tried, nodes = self.collect_fixture(priority=[held], held=held, priority_only=True)
        self.assertEqual(report['device_probe_scope'], 'FULL_FLEET_FALLBACK')
        self.assertEqual(report['deferred_device_probe_nodes'], [])
        self.assertEqual(tried, nodes - {held})


if __name__ == '__main__':
    unittest.main()
