"""Regression checks for 640-card Slurm placement observations."""
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from read_training_slurm_membership import allocated_dcu, dispatch_candidates, group_coverage, numbered_step_usage, snapshot
from tuning_queue import TRAINING80_LEDGER, sha, training80_inventory


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_bytes())


def step(members, count):
    return {"member_nodes": members, "allocated_DCU": count}


class StepUsageTest(unittest.TestCase):
    def test_device_probe_gives_slurm_a_bounded_term_grace(self):
        from collect_training80_dispatch_snapshot import PROBE_TERM_GRACE_S, run_device_probe

        argv = ['srun', '--threads=1']
        for needs_kill in (False, True):
            with self.subTest(needs_kill=needs_kill), patch(
                    'collect_training80_dispatch_snapshot.subprocess.Popen') as start:
                process = start.return_value
                expiration = subprocess.TimeoutExpired(argv, 5)
                after_term = subprocess.TimeoutExpired(argv, PROBE_TERM_GRACE_S) if needs_kill else (b'partial', b'')
                process.communicate.side_effect = [expiration, after_term]
                with self.assertRaises(subprocess.TimeoutExpired) as raised:
                    run_device_probe(argv, capture_output=True, timeout=5, check=False)
                self.assertIs(raised.exception, expiration)
                process.terminate.assert_called_once()
                self.assertEqual(process.communicate.call_args.kwargs, {'timeout': PROBE_TERM_GRACE_S})
                self.assertEqual(process.kill.call_count, int(needs_kill))
                self.assertEqual(process.wait.call_count, int(needs_kill))
                process.stdout.close.assert_called_once()
                process.stderr.close.assert_called_once()

    def test_cpu_device_check_reads_kfd_holders_and_requested_socket_ports(self):
        from collect_training80_dispatch_snapshot import device_status

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            drm, proc, kfd = (root / part for part in ('drm', 'proc', 'kfd'))
            (drm / 'renderD128').mkdir(parents=True)
            (proc / 'net').mkdir(parents=True)
            (kfd / '554').mkdir(parents=True)
            header = 'sl local_address remote_address st\n'
            for table in ('tcp', 'tcp6', 'udp', 'udp6'):
                (proc / 'net' / table).write_text(header, encoding='ascii')
            (proc / 'net' / 'udp6').write_text(header + '0: 000000000:002A 000000000:0000 07\n', encoding='ascii')
            with patch('collect_training80_dispatch_snapshot.DRM', drm), \
                 patch('collect_training80_dispatch_snapshot.PROC', proc), \
                 patch('collect_training80_dispatch_snapshot.KFD_PROCESSES', kfd), \
                 patch('pathlib.Path.resolve', return_value=Path('0000:20:00.0')):
                result = device_status([42, 512])
                self.assertEqual(result['PCIs'], ['0000:20:00.0'])
                self.assertEqual(result['kfd_process_count'], 1)
                self.assertEqual(result['occupied_requested_ports'], [42])
                (proc / 'net' / 'udp6').write_text('unrecognized heading\n', encoding='ascii')
                with self.assertRaisesRegex(ValueError, 'socket table format'):
                    device_status([42])

    def test_cpu_and_unknown_gres_are_distinct(self):
        self.assertEqual(allocated_dcu("cpu=4,mem=8G,node=1"), 0)
        self.assertIsNone(allocated_dcu("(null)"))
        self.assertIsNone(allocated_dcu("cpu=4,gres/gpu=2"))
        self.assertEqual(allocated_dcu("cpu=8,gres/dcu=8,gres/dcu:bw=8"), 8)
        self.assertEqual(allocated_dcu("gres/dcu:bw=3,gres/dcu:other=2"), 5)
        for value in ("gres/dcu=4,gres/dcu:bw=8", "gres/dcu=bogus", "gres/dcu=3,gres/dcu=3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                allocated_dcu(value)

    def test_overlap_never_double_counts_and_missing_devices_stay_unknown(self):
        usage = numbered_step_usage(
            {"a": step(["n1", "n2"], 10), "b": step(["n2"], 1), "c": step(["n3"], None),
             "d": step(["n4"], 0)}, {"n1", "n2", "n3", "n4", "n5"}, 8)
        self.assertEqual((usage["minimum_assigned_DCU"], usage["maximum_assigned_DCU"]), (10, 19))
        self.assertEqual(usage["nodes_without_numbered_DCU_step"], ["n4", "n5"])
        self.assertEqual(usage["nodes_with_only_CPU_numbered_steps"], ["n4"])
        self.assertEqual(usage["unknown_DCU_allocation_nodes"], ["n3"])
        self.assertEqual(usage["assigned_DCU_bounds_by_node"], {
            "n1": [2, 8], "n2": [2, 8], "n3": [0, 8], "n4": [0, 0], "n5": [0, 0]})

    def test_bridge_unions_two_overlapping_components(self):
        usage = numbered_step_usage(
            {"a": step(["n1"], 6), "b": step(["n2"], 5), "c": step(["n1", "n2"], 4)},
            {"n1", "n2"}, 8)
        self.assertEqual((usage["minimum_assigned_DCU"], usage["maximum_assigned_DCU"]), (11, 15))

    def test_foreign_node_makes_parent_totals_unavailable(self):
        for count in (0, 8):
            with self.subTest(count=count):
                result = numbered_step_usage({"a": step(["other"], count)}, {"n1", "n2"}, 8)
                self.assertIsNone(result["maximum_assigned_DCU"])
                self.assertIsNone(result["nodes_without_numbered_DCU_step"])

    def test_group_coverage_distinguishes_whole_node_from_partial_or_uncertain(self):
        expected = {f"n{i}": "T01" if i <= 3 else "O01" for i in range(1, 8)}
        usage = numbered_step_usage({
            "a": step(["n1"], 8), "b": step(["n2"], 3), "c": step(["n3"], None),
            "d": step(["n4"], 0), "e": step(["n6", "n7"], 9),
        }, set(expected), 8)
        report = group_coverage(expected, usage, cards_per_node=8, valid_parent=True)
        self.assertEqual({node: row["state"] for group in report.values() for node, row in group.items()}, {
            "n1": "FULLY_ASSIGNED", "n2": "PARTIALLY_ASSIGNED", "n3": "UNREPORTED_DCU_ALLOCATION",
            "n4": "CPU_STEP_ONLY", "n5": "NO_NUMBERED_STEP", "n6": "ASSIGNMENT_UNCERTAIN",
            "n7": "ASSIGNMENT_UNCERTAIN",
        })
        stale = group_coverage(expected, usage, cards_per_node=8, valid_parent=False)
        self.assertTrue(all(row == {"numbered_step_assigned_DCU_bounds": None, "state": "UNVERIFIED_PARENT"}
                            for group in stale.values() for row in group.values()))

    def test_dispatch_order_skips_unknown_fully_assigned_and_unverified_nodes(self):
        nodes = {f'n{i}' for i in range(1, 8)}
        usage = numbered_step_usage({
            'a': step(['n1'], 6), 'b': step(['n2'], 2), 'c': step(['n3'], None),
            'd': step(['n4'], 0), 'e': step(['n5'], 8), 'f': step(['n7'], 8),
        }, nodes, 8)
        expected = {node: 'T01' for node in nodes}
        report = dispatch_candidates({
            'p1': {'coverage_by_group_and_node': group_coverage(expected, usage, 8, True)},
            'p2': {'coverage_by_group_and_node': group_coverage({'other': 'T02'}, usage, 8, False)},
        })
        self.assertEqual([row['node'] for row in report['ordered_node_checks']], ['n6', 'n4', 'n2', 'n1'])
        self.assertEqual(report['nodes_by_state']['UNVERIFIED_PARENT'], 1)
        self.assertFalse(report['physical_admission_checked'])


class RealPlacementFixtureTest(unittest.TestCase):
    def setUp(self):
        ledger = load("coordination/training/training80-active-node-task-map-v1.json")
        self.groups = {job["JobId"]: {name: row["allocations"][job["JobId"]]
                                      for name, row in ledger["groups"].items()}
                       for job in ledger["jobs"]}
        self.hostlists = {job["NodeList"]: [node for members in self.groups[job["JobId"]].values()
                                            for node in members] for job in ledger["jobs"]}
        self.replies = {}
        self.rows = {}
        for job in ledger["jobs"]:
            parent, members = job["JobId"], job["NodeList"]
            self.replies[("squeue", "-h", "-j", parent, "-o", "%i|%T|%D|%N")] = [
                f"{parent}|RUNNING|{job['NumNodes']}|{members}"]
            self.replies[("scontrol", "show", "job", parent, "--oneliner")] = [
                f"JobId={parent} JobState=RUNNING UserId=fixture(1234) AllocTRES={job['AllocTRES']}"]
            self.rows[parent] = [f"StepId={parent}.extern TRES=(null)"]

        fleet = load("results/bw1000/2026-11-20/GLM-TP16-SHCA-Ring-Tree-560-collector-terminal.json")
        placement = load("code/bw1000/fleet-glm-tp16-shca-ring-tree-wide-v2/PLACEMENT.json")["wide"]
        # Use the two real expanded placements and the separate SDXL node from the overlapping run window.
        for parent, handle in (("1087656", "1087656.542"), ("1103606", "1103606.468")):
            row, = [row for row in fleet["accounting"][parent] if row[0] == handle]
            self.add_step(parent, row, [node for pair in placement[parent] for node in pair])
        sdxl = load("results/bw1000/2026-11-20/training-sdxl-original540-frozen-conv-dgrad-fp32-terminal.json")
        native, = [row for row in sdxl["receipts"]["rows"] if row[0] == "1087656.540"]
        self.add_step("1087656", native, [native[5]])

    def add_step(self, parent, accounting, members):
        handle, expression, tres = accounting[0], accounting[5], accounting[6]
        self.hostlists[expression] = members
        self.rows[parent].append(
            f"StepId={handle} State=RUNNING Nodes={len(members)} NodeList={expression} TRES={tres}")

    def query(self, *args):
        if args[:3] == ("scontrol", "show", "hostnames"):
            return self.hostlists[args[3]]
        if args[:3] == ("scontrol", "show", "step"):
            return self.rows[args[3]]
        return self.replies[args]

    def test_reconstructs_561_numbered_step_cards_from_distinct_nodes(self):
        with patch("read_training_slurm_membership.query", side_effect=self.query):
            report = snapshot(self.groups)
        self.assertEqual(report["allocated_DCU"], 640)
        self.assertEqual((report["minimum_numbered_step_assigned_DCU"],
                          report["maximum_numbered_step_assigned_DCU"]), (561, 561))
        self.assertIsNone(report["busy_measurement"])
        usage = report["parents"]["1087656"]["numbered_step_usage"]
        self.assertEqual(usage["assigned_DCU_bounds_by_node"]["f12r3n20"], [1, 1])
        by_node = {node: row for parent in report["parents"].values()
                   for group in parent["coverage_by_group_and_node"].values() for node, row in group.items()}
        self.assertEqual(len(by_node), 80)
        self.assertEqual(sum(row['state'] == 'FULLY_ASSIGNED' for row in by_node.values()), 70)
        self.assertEqual(sum(row['state'] == 'NO_NUMBERED_STEP' for row in by_node.values()), 9)
        self.assertEqual(by_node["f12r3n20"]["state"], "PARTIALLY_ASSIGNED")
        checks = report['dispatch_candidates']['ordered_node_checks']
        self.assertEqual([row['state'] for row in checks], ['NO_NUMBERED_STEP'] * 9 + ['PARTIALLY_ASSIGNED'])
        self.assertEqual(checks[-1], {'parent': '1087656', 'group': 'O01', 'node': 'f12r3n20',
                                      'state': 'PARTIALLY_ASSIGNED', 'numbered_step_assigned_DCU_bounds': [1, 1]})

    def test_parent_with_only_extern_keeps_all_nodes_visible(self):
        self.rows = {parent: [f"StepId={parent}.extern TRES=(null)"] for parent in self.rows}
        with patch("read_training_slurm_membership.query", side_effect=self.query):
            report = snapshot(self.groups)
        group_rows = [row for parent in report["parents"].values()
                      for group in parent["coverage_by_group_and_node"].values() for row in group.values()]
        self.assertEqual(len(group_rows), 80)
        self.assertTrue(all(row['state'] == 'NO_NUMBERED_STEP' for row in group_rows))
        self.assertEqual(report["maximum_numbered_step_assigned_DCU"], 0)
        self.assertEqual(report["maximum_running_numbered_step_assigned_DCU"], 0)
        self.assertIsNone(report["busy_measurement"])

    def test_nonrunning_records_keep_placement_protection_without_inflating_running_count(self):
        parent = '1087656'
        self.rows[parent] = [row.replace('State=RUNNING', 'State=TIMEOUT')
                             if row.startswith('StepId=' + parent + '.540 ') else row for row in self.rows[parent]]
        with patch('read_training_slurm_membership.query', side_effect=self.query):
            report = snapshot(self.groups)
        self.assertEqual((report['minimum_running_numbered_step_assigned_DCU'],
                          report['maximum_running_numbered_step_assigned_DCU']), (560, 560))
        self.assertEqual(report['minimum_numbered_step_assigned_DCU'], 561)
        entry = report['parents'][parent]
        self.assertEqual(entry['active_steps'][parent + '.540']['state'], 'TIMEOUT')
        self.assertIn('f12r3n20', entry['running_numbered_step_usage']['nodes_without_numbered_DCU_step'])
        self.assertEqual(entry['coverage_by_group_and_node']['O01']['f12r3n20']['state'], 'PARTIALLY_ASSIGNED')

    def test_queue_can_consider_clear_nodes_during_known_multi_group_steps(self):
        with patch('read_training_slurm_membership.query', side_effect=self.query):
            report = snapshot(self.groups, owner_uid=1234)
        self.assertFalse(report['membership_qualified'])
        self.assertTrue(all(row['step_locations_exact'] for row in report['parents'].values()))
        report['ledger_SHA256'] = sha(TRAINING80_LEDGER)
        nodes = {node: {'zone': zone, 'parent': parent}
                 for parent, groups in self.groups.items()
                 for zone, names in groups.items() for node in names}
        placements = training80_inventory({'membership': report, 'nodes': nodes})
        candidates = {node for node, (_, _, slurm_clear) in placements.items() if slurm_clear}
        self.assertEqual(candidates, {row['node'] for row in report['dispatch_candidates']['ordered_node_checks']
                                      if row['state'] == 'NO_NUMBERED_STEP'})
        self.assertEqual(len(candidates), 9)
        self.assertNotIn('f12r3n20', candidates)

        # A placed process takes precedence over the redundant coverage label.
        member = 'f12r3n20'
        report['parents']['1087656']['coverage_by_group_and_node']['O01'][member] = {
            'state': 'NO_NUMBERED_STEP', 'numbered_step_assigned_DCU_bounds': [0, 0]}
        self.assertFalse(training80_inventory({'membership': report, 'nodes': nodes})[member][2])

    def test_job_owner_and_outside_node_mark_parent_unqualified_for_placement(self):
        parent = '1087656'
        self.add_step(parent, (parent+'.999', '', '', '', '', 'foreign-node', 'cpu=1,gres/dcu=1'),
                      ['foreign-node'])
        with patch('read_training_slurm_membership.query', side_effect=self.query):
            report = snapshot(self.groups, owner_uid=4321)
        self.assertFalse(report['parents'][parent]['step_locations_exact'])
        self.assertFalse(report['parents'][parent]['owned_by_observer'])
        self.assertIsNone(report['maximum_numbered_step_assigned_DCU'])

    def test_one_shot_sends_only_unoccupied_nodes_and_preserves_failed_checks(self):
        from collect_training80_dispatch_snapshot import collect, receive

        ledger_raw = TRAINING80_LEDGER.read_bytes()
        pins = {node: [f'0000:{bus:02x}:00.0' for bus in range(8)]
                for groups in self.groups.values() for nodes in groups.values() for node in nodes}
        request = 'f' * 32
        initial = snapshot(self.groups, owner_uid=1234, run_query=self.query)
        expected = {job: [] for job in self.groups}
        for row in initial['dispatch_candidates']['ordered_node_checks']:
            if row['state'] == 'NO_NUMBERED_STEP':
                expected[row['parent']].append(row['node'])
        candidates = [node for nodes in expected.values() for node in nodes]
        self.assertEqual(len(candidates), 9)
        missing, unavailable, held, conflict, race, *clear = candidates
        calls = []

        def run(args, *, capture_output, timeout, check):
            self.assertTrue(capture_output)
            self.assertGreater(timeout, 0)
            if args[0] != 'srun':
                return subprocess.CompletedProcess(args, 0, '\n'.join(self.query(*args)).encode(), b'')
            parent = next(arg.split('=', 1)[1] for arg in args if arg.startswith('--jobid='))
            nodes = next(arg.split('=', 1)[1] for arg in args if arg.startswith('--nodelist=')).split(',')
            self.assertEqual(nodes, expected[parent])
            self.assertIn('--gres=none', args)
            self.assertNotIn('--overlap', args)
            self.assertFalse(check)
            calls.append(parent)
            if race in nodes:
                self.add_step(parent, (parent+'.998', '', '', '', '', race, 'cpu=1,gres/dcu=8'), [race])
            rows = []
            for node in nodes:
                if node == missing:
                    continue
                row = dict(request_id=args[-3], parent=parent, step=parent+'.997', node=node,
                           ports_checked=[43000], observer_uid=1234)
                if node == unavailable:
                    row['probe_error'] = 'PermissionError'
                else:
                    row.update(PCIs=pins[node], kfd_process_count=int(node == held),
                               own_device_fd_process_count=0, occupied_requested_ports=[43000] if node == conflict else [])
                rows.append(json.dumps(row).encode())
            return subprocess.CompletedProcess(args, int(missing in nodes), b'\n'.join(rows), b'')

        started = time.monotonic()
        with patch('collect_training80_dispatch_snapshot.os.getuid', return_value=1234, create=True):
            report = collect(request, ledger_raw, pins, [43000], run=run)
        self.assertEqual(sorted(calls), sorted(parent for parent, nodes in expected.items() if nodes))
        self.assertEqual(len(report['nodes']), 80)
        self.assertFalse(report['nodes'][missing]['device_probe_observed'])
        self.assertEqual(report['nodes'][missing]['probe_error'], 'ROW_MISSING')
        self.assertEqual(report['nodes'][unavailable]['probe_error'], 'PermissionError')
        self.assertFalse(report['nodes'][held]['KFD_clean'])
        self.assertEqual(report['nodes'][conflict]['occupied_ports'], [43000])
        self.assertTrue(report['nodes'][race]['active_steps'])
        self.assertTrue(all(report['nodes'][n]['ports_clean'] for n in clear))
        self.assertIsNone(report['measured_busy_percent'])
        raw = json.dumps(report).encode()
        accepted = receive(raw, started, request)
        self.assertEqual(accepted['local_receipt']['request_id'], request)
        placements = training80_inventory(accepted)
        self.assertFalse(placements[race][2])
        self.assertTrue(all(placements[node][2] for node in clear))
        with self.assertRaisesRegex(ValueError, 'wrong dispatch snapshot request'):
            receive(raw, started, '0'*32)

    def test_full_pool_cpu_probe_isolates_one_unavailable_batch_and_rejects_unbound_rows(self):
        from collect_training80_dispatch_snapshot import NODES_PER_PROBE, collect, receive

        for parent in self.groups:
            self.rows[parent] = [f'StepId={parent}.extern TRES=(null)']
            self.replies[('scontrol', 'show', 'step', parent, '--oneliner')] = self.rows[parent]
        ledger_raw = TRAINING80_LEDGER.read_bytes()
        pins = {node: [f'0000:{bus:02x}:00.0' for bus in range(8)]
                for groups in self.groups.values() for nodes in groups.values() for node in nodes}
        initial = snapshot(self.groups, owner_uid=1234, run_query=self.query)
        missing = {parent: [] for parent in self.groups}
        for row in initial['dispatch_candidates']['ordered_node_checks']:
            missing[row['parent']].append(row['node'])
        first, second = sorted(missing)
        bad = missing[first][0]
        foreign = missing[second][0]
        request = '8' * 32
        witnessed = []

        def run(args, *, capture_output, timeout, check):
            if args[0] != 'srun':
                return subprocess.CompletedProcess(args, 0, '\n'.join(self.query(*args)).encode(), b'')
            nodes = next(arg.split('=', 1)[1] for arg in args if arg.startswith('--nodelist=')).split(',')
            parent, batch_request = args[-2], args[-3]
            self.assertLessEqual(len(nodes), NODES_PER_PROBE)
            self.assertLessEqual(timeout, 10)
            self.assertIn('--gres=none', args)
            witnessed.append((parent, batch_request, nodes))
            if bad in nodes:
                return subprocess.CompletedProcess(args, 1, b'', b'Job step creation temporarily disabled')
            rows = [dict(request_id=request if node == foreign else batch_request, parent=parent,
                         step=parent + '.997', node=node, ports_checked=[], observer_uid=1234, PCIs=pins[node],
                         kfd_process_count=0, own_device_fd_process_count=0, occupied_requested_ports=[])
                    for node in nodes]
            return subprocess.CompletedProcess(args, 0, '\n'.join(map(json.dumps, rows)).encode(), b'')

        started = time.monotonic()
        with patch('collect_training80_dispatch_snapshot.os.getuid', return_value=1234, create=True):
            report = collect(request, ledger_raw, pins, [], run=run)
        self.assertEqual(len(witnessed), 10)
        self.assertEqual(len({batch_request for _, batch_request, _ in witnessed}), 10)
        tried = [node for _, _, nodes in witnessed for node in nodes]
        self.assertEqual(len(tried), 80)
        self.assertEqual(set(tried), set(pins))
        rejected = set(missing[first][:8] + missing[second][:8])
        self.assertEqual({name for name, row in report['nodes'].items() if not row['device_probe_observed']}, rejected)
        self.assertEqual(report['nodes'][bad]['probe_error'], 'SRUN_FAILED_WITHOUT_ROWS')
        self.assertEqual(report['nodes'][foreign]['probe_error'], 'ValueError')
        self.assertEqual(sum(len(row['received_nodes']) for row in report['probes'].values()), 64)
        accepted = receive(json.dumps(report).encode(), started, request)
        self.assertEqual({name for name, row in accepted['nodes'].items() if row['KFD_clean'] and row['PID_clean']},
                         set(pins) - rejected)

    def test_foreign_or_unrecognized_steps_are_not_silently_ignored(self):
        self.rows["1087656"].append("StepId=other.800 State=RUNNING")
        with patch("read_training_slurm_membership.query", side_effect=self.query):
            with self.assertRaisesRegex(ValueError, "unrecognized live step"):
                snapshot(self.groups)


if __name__ == "__main__":
    unittest.main()
