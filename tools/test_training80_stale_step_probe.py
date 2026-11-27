"""Replay the four frozen timeout identities through the production probe entry."""
import copy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from collect_training80_dispatch_snapshot import collect, receive
from report_training80_backlog import fleet_view
from training_dispatch_allocation import REPLACEMENT_LEDGER, STALE_STEP_PROBE_SCOPE, allocation
from tuning_queue import Queue, training80_inventory


REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / ('results/bw1000/2026-11-27/'
                   'training80-2106-refill4571-eight-computing-and-four-controller-accounting-conflicts.json')
REQUEST = '3' * 32


class ProductionProbeTest(unittest.TestCase):
    def collect(self, *, held=(), initial_error=None, final_error=None):
        raw = REPLACEMENT_LEDGER.read_bytes()
        _, groups, owners = allocation(raw)
        pins = {node: [f'0000:{index:02x}:00.0' for index in range(8)] for node in owners}
        frozen = json.loads(EVIDENCE.read_bytes())['stale_accounting_vs_controller']
        self.assertEqual(set(STALE_STEP_PROBE_SCOPE), set(frozen))
        for handle, (node, label, tres) in STALE_STEP_PROBE_SCOPE.items():
            self.assertEqual(frozen[handle]['scontrol_match'],
                             [dict(StepId=handle, Name=label, NodeList=node, State='TIMEOUT', TRES=tres)])
        batches = []
        count = 0
        exact_reads = []

        def members(parent):
            return sorted(node for zone in groups[parent].values() for node in zone)

        def step_line(handle):
            fields = frozen[handle]['scontrol_match'][0]
            return ' '.join(f'{key}={value}' for key, value in fields.items()) + ' Nodes=1'

        def run(argv, *, capture_output, timeout, check):
            nonlocal count
            self.assertTrue(capture_output)
            self.assertGreater(timeout, 0)
            args = tuple(argv)
            lines = []
            err = b''
            if args[0] == 'srun':
                self.assertFalse(check)
                self.assertIn('--gres=none', args)
                self.assertIn('--nodes=1', args)
                self.assertNotIn('--overlap', args)
                node, = [arg.split('=', 1)[1] for arg in args if arg.startswith('--nodelist=')]
                parent = owners[node][0]
                batches.append(node)
                lines.append(json.dumps(dict(node=node, parent=parent, request_id=args[-3], step=parent+'.9991',
                                             observer_uid=1234, ports_checked=[43000], PCIs=pins[node],
                                             kfd_process_count=0, own_device_fd_process_count=0,
                                             occupied_requested_ports=[])))
            elif args[:3] == ('squeue', '-h', '-j'):
                parent = args[3]
                lines = [f'{parent}|RUNNING|40|{parent}_members']
                if parent == sorted(groups)[0]:
                    count += 1
            elif args[:3] == ('scontrol', 'show', 'hostnames'):
                lines = members(args[3].removesuffix('_members')) if args[3].endswith('_members') else [args[3]]
            elif args[:3] == ('scontrol', 'show', 'job'):
                lines = [f'JobId={args[3]} UserId=observer(1234) JobState=RUNNING AllocTRES=gres/dcu=320']
            elif args[:3] == ('scontrol', 'show', 'step'):
                key = args[3]
                if '.' not in key:
                    lines = [step_line(handle) for handle in frozen if handle.startswith(key+'.')]
                    if count == 2 and final_error == 'other-step' and key == '1194900':
                        lines.append('StepId=1194900.9996 Name=other NodeList=f13r4n03 Nodes=1 State=RUNNING TRES=cpu=1,node=1')
                else:
                    exact_reads.append((count, key))
                    lines = [step_line(key)]
                    if key == '1103606.1644' and count == 1 and initial_error == 'wrong-name':
                        lines = [lines[0].replace('af719c997048', 'wrong')]
                    if key == '1103606.1644' and count == 1 and initial_error == 'duplicate':
                        lines *= 2
            elif args[:4] == ('squeue', '--steps', '-h', '-j'):
                self.assertEqual(args[5:], ('-o', '%i|%j|%N'))
                if initial_error == 'queue-live' and count == 1 and args[4] == '1103606':
                    lines = ['1103606.888|other|f13r4n06']
            elif args[:3] == ('sstat', '-nP', '-j'):
                self.assertEqual(args[4:], ('-o', 'JobID,NTasks,MaxRSS'))
                if args[3] == '1103606.1644' and count == 1 and initial_error == 'stat-stderr':
                    err = b'warning'
                if args[3] == '1103606.1644' and count == 2 and final_error == 'stat-live':
                    lines = ['1103606.1644|1|100K']
            else:
                raise AssertionError(args)
            return subprocess.CompletedProcess(argv, 0, ('\n'.join(lines)).encode(), err)

        started = time.monotonic()
        with patch('collect_training80_dispatch_snapshot.os.getuid', return_value=1234, create=True):
            report = collect(REQUEST, raw, pins, [43000], held_nodes=held, stale_step_probe_only=True, run=run)
        received = receive(json.dumps(report).encode(), started, REQUEST)
        self.assertCountEqual(exact_reads, [(phase, handle) for phase in (1, 2) for handle in frozen])
        return received, batches

    def test_positive_probe_cannot_reserve_or_change_raw(self):
        evidence, batches = self.collect()
        selected = {scope[0] for scope in STALE_STEP_PROBE_SCOPE.values()}
        self.assertCountEqual(batches, selected)
        self.assertEqual(evidence['membership']['distinct_nodes'], 80)
        for handle, (node, _, _) in STALE_STEP_PROBE_SCOPE.items():
            parent = evidence['membership']['parents'][handle.split('.')[0]]
            self.assertEqual(parent['active_steps'][handle]['state'], 'TIMEOUT')
            self.assertEqual(evidence['nodes'][node]['active_steps'], [handle])
            self.assertTrue(evidence['stale_step_probe_only']['nodes'][node]['physical_probe_witness_complete'])
        placements = training80_inventory(evidence)
        self.assertTrue(all(not placements[node][2] for node in selected))
        _, rows, fresh = fleet_view(evidence, time.monotonic(), time.time()-time.monotonic(), REPLACEMENT_LEDGER)
        self.assertTrue(fresh)
        self.assertEqual({node for node, row in rows.items() if row['admission'] == 'physical_probe_only'}, selected)
        self.assertFalse(any(row['admission'] == 'clear' for row in rows.values()))
        with TemporaryDirectory() as temporary:
            queue = Queue(Path(temporary) / 'empty.sqlite')
            try:
                plan = queue.plan(evidence)
            finally:
                queue.db.close()
        self.assertEqual(plan['jobs'], [])
        self.assertEqual({node for zone in plan['node_coverage'].values() for parent in zone.values()
                          for node in parent.get('physical_probe_only', [])}, selected)

        forged = copy.deepcopy(evidence)
        forged['stale_step_probe_only']['nodes']['f13r4n06']['final']['queue_clear'] = False
        with self.assertRaisesRegex(ValueError, 'completion differs'):
            training80_inventory(forged)

    def test_inherited_overlap_stops_before_queries(self):
        with patch.dict('collect_training80_dispatch_snapshot.os.environ', {'SLURM_OVERLAP':'1'}):
            with self.assertRaisesRegex(ValueError, 'forbids inherited'):
                self.collect()

    def test_initial_conflict_prevents_target_probe(self):
        for case in ('wrong-name', 'duplicate', 'queue-live', 'stat-stderr'):
            with self.subTest(case=case):
                report, batches = self.collect(initial_error=case, held=('f13r4n09',))
                self.assertNotIn('f13r4n06', batches)
                self.assertNotIn('f13r4n09', batches)
                self.assertFalse(report['stale_step_probe_only']['nodes']['f13r4n06']['probe_eligible'])

    def test_final_conflict_retains_probe_and_blocks_promotion(self):
        for case, node in (('stat-live', 'f13r4n06'), ('other-step', 'f13r4n03')):
            with self.subTest(case=case):
                report, batches = self.collect(final_error=case)
                self.assertIn(node, batches)
                self.assertTrue(report['nodes'][node]['device_probe_observed'])
                self.assertFalse(report['stale_step_probe_only']['nodes'][node]['physical_probe_witness_complete'])
                self.assertFalse(training80_inventory(report)[node][2])


if __name__ == '__main__':
    unittest.main()
