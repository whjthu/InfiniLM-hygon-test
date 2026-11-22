"""Safety and decision invariants for expensive-run orchestration (no GPU/SSH)."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from tuning_queue import TRAINING80_LEDGER,Queue,canonical,digest,expand,refit_provisional_loans,select_next_rung,receive_snapshot,sha
ROOT=Path(__file__).resolve().parents[1]

class TuningTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'q.sqlite3';self.q=Queue(self.path);self.addCleanup(self.q.db.close)
        self.campaign=json.loads((ROOT/'coordination/tuning/bw1000-campaign-v1.json').read_bytes())
        self.rows=expand(self.campaign);self.q.seed(self.rows)
        self.fleet=dict(observed_epoch=time.time(),complete=True,total_cards=136,measured_busy_percent=0,nodes={})
        for i in range(17):
            zone='GLM' if i<4 else 'DeepSeek' if i<12 else 'Kimi' if i<16 else 'control'
            self.fleet['nodes']['node'+str(i)]=dict(zone=zone,PCIs=['PCI'+str(j)for j in range(8)],KFD_clean=True,PID_clean=True,ports_clean=True,membership_valid=True,active_steps=[])

        self.fleet=receive_snapshot(self.fleet,time.monotonic(),'test-request','a'*64)

    def ready(self, suffix='', zone='GLM', count=2):
        spec=next(r for r in self.rows if r['model']==zone)
        b=dict(zone=zone,nodes_required=count,allowed_nodes=[n for n,v in self.fleet['nodes'].items()if v['zone']==zone],publication_commit='a'*40)
        ident=digest([suffix,b]);self.q.db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?)',
            (ident,spec['id'],'model','READY',canonical(b).decode(),'[]',None,None,time.time()))
        return ident

    def result(self):
        return dict(request_id='request1',intent='original-intent',terminal=True,post_clean=True,
                    evidence_SHA256='e'*64,planned=20,completed=12,failed=0,rejected=3,canceled=2,missing=3,
                    qualification='model',keeper_passed=False,passed=False,promote=False)

    def active(self, unknown=False):
        ident=self.ready();plan=self.q.plan(self.fleet);self.q.reserve(plan,self.fleet)
        self.q.submission(ident,dict(request_id='request1',intent='original-intent'),unknown)
        return ident

    def comparison(self,scope):
        return dict(model='GLM',hardware='same',precision='BF16',scope=scope,workload_SHA256='w'*64,weights_SHA256='m'*64,metric='visible_per_user_decode_tps')

    def test_prepare_exact_published_bundle_and_no_replay(self):
        import hashlib
        spec=self.rows[0];repo=Path(self.tmp.name)/'repo';artifacts=[]
        for role in ['source','deps','claim','launcher']:
            name='code/bw1000/test/'+role;p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(role.encode())
            artifacts.append(dict(role=role,path=name,SHA256=hashlib.sha256(role.encode()).hexdigest()))
        binding=dict(root_reviewed=True,candidate=spec['id'],stage='component',rung=0,workload_SHA256=digest(spec['workload']),memory_floor_bytes=spec['memory_floor_bytes'],deadline_s=600,nodes_required=spec['nodes_required'],zone=spec['zone'],prerequisites_closed=spec['prerequisites'],prerequisite_evidence=['exact-prior-receipt'],artifacts=artifacts,publication_commit='a'*40,claim_semantics_checked=True,inputoutput_binding_SHA256='i'*64,dependency_binding_SHA256='d'*64)
        def git(argv,**kw):
            return b'' if argv[1]=='merge-base' else (repo/argv[2].split(':',1)[1]).read_bytes()
        with patch('tuning_queue.subprocess.check_output',side_effect=git):
            ident=self.q.prepare(binding,repo)
            with self.assertRaisesRegex(ValueError,'already attempted'):self.q.prepare(binding,repo)
            changed=copy.deepcopy(binding);changed['memory_floor_bytes']-=1
            with self.assertRaisesRegex(ValueError,'floor drift'):self.q.prepare(changed,repo)
        self.assertEqual(self.q.summary()['runs'],{'READY':1})

    def test_grid_deterministic_bounded_and_deduplicated(self):
        self.assertEqual(len(self.rows),47)
        self.assertEqual(self.rows,expand(self.campaign))
        self.q.seed(self.rows);self.assertEqual(self.q.summary()['candidates'],47)
        self.assertEqual(self.q.summary()['runs'],{})
        x=copy.deepcopy(self.campaign);x['families'][1]['max_candidates']=1
        with self.assertRaisesRegex(ValueError,'space exceeds'):expand(x)

    def test_semantic_dedup_excludes_display_family_name(self):
        x=copy.deepcopy(self.campaign);x['families'][0]['id']='renamed'
        self.assertEqual([r['id']for r in self.rows],[r['id']for r in expand(x)])

    def test_stale_inventory_rejected(self):
        self.fleet['local_receipt']['received_monotonic']-=61
        with self.assertRaisesRegex(ValueError,'fresh'):self.q.plan(self.fleet)

    def test_operator_admission_retirement_requires_the_original_proof(self):
        from prepare_training80_dispatch_handoff import reserve_handoff

        run = 'operator-admission-1'
        intent = 'local-only/original-op/intent.json'
        packet = dict(request='a' * 32, case='O01.operator-pair', kind='operator',
                      workload_SHA256='b' * 64, placement_by_parent={'1103606': {'node0': 1}},
                      handoff_SHA256='c' * 64, snapshot_payload_SHA256='d' * 64, invocation=None,
                      launcher='operator.py', expires_local_monotonic=time.monotonic() + 30)
        claimed = reserve_handoff(self.q, run, packet, intent)
        self.q.submission(run, claimed['handle'], unknown=True)
        proof = dict(request_id=packet['request'], intent=intent, failure_phase='OPERATOR_PRE_SLURM_ADMISSION',
                     launch_attempted=False, matching_accounting_rows=0, post_clean=True, node='node0',
                     evidence_SHA256='e' * 64, transport_returncode=1, remote_intent_absent=True,
                     admission_SHA256=packet['snapshot_payload_SHA256'])
        for changed in (dict(remote_intent_absent=False), dict(admission_SHA256='f' * 64),
                        dict(matching_accounting_rows=1)):
            with self.assertRaises(ValueError):
                self.q.retire_remote_prelaunch_direct(run, {**proof, **changed})
        self.assertEqual(self.q.summary()['runs'], {'UNKNOWN': 1})
        self.q.retire_remote_prelaunch_direct(run, proof)
        self.assertEqual(self.q.summary()['runs'], {'RETIRED_PRELAUNCH': 1})


    def test_unknown_and_expired_intent_still_reserves_nodes(self):
        self.active(True);self.ready('second');self.ready('third')
        # Advancing a reservation age does not release original ownership.
        self.q.db.execute('UPDATE runs SET created=0 WHERE state="UNKNOWN"')
        p=self.q.plan(self.fleet);self.assertEqual(len(p['jobs']),1)
        self.assertEqual(self.q.summary()['runs']['UNKNOWN'],1)

    def test_plan_reserves_disjoint_zones_backfill(self):
        for i in range(4):self.ready(str(i))
        self.ready('deep','DeepSeek',1);self.ready('kimi','Kimi',1)
        p=self.q.plan(self.fleet)
        self.assertEqual(len(p['jobs']),4)
        nodes=[n for r in p['jobs']for n in r['nodes']]
        self.assertEqual(len(nodes),len(set(nodes)))
        self.assertEqual(p['selected_cards'],48)
        self.assertEqual(p['busy_percent'],0)
        self.assertFalse(p['busy_is_allocation'])

    def test_snapshot_conflict_blocks_busy_node(self):
        self.ready();self.fleet['nodes']['node0']['active_steps']=['original.1']
        self.fleet['nodes']['node1']['KFD_clean']=False
        self.fleet['local_receipt']['payload_SHA256']=digest({k:v for k,v in self.fleet.items()if k!='local_receipt'})
        self.assertEqual(self.q.plan(self.fleet)['jobs'][0]['nodes'],['node2','node3'])

    def training80_payload(self):
        ledger = json.loads(TRAINING80_LEDGER.read_bytes())
        parents = {}
        nodes = {}
        for job in ledger['jobs']:
            parent = job['JobId']
            coverage = {}
            parents[parent] = dict(state='RUNNING', nodes=40, allocated_DCU=320, ledger_exact=True,
                                   owned_by_observer=True, step_locations_exact=True, active_steps={},
                                   coverage_by_group_and_node=coverage)
            for zone, group in ledger['groups'].items():
                coverage[zone] = {}
                for name in group['allocations'][parent]:
                    coverage[zone][name] = dict(state='NO_NUMBERED_STEP',
                                               numbered_step_assigned_DCU_bounds=[0, 0])
                    nodes[name] = dict(parent=parent, zone=zone,
                                       PCIs=[f'0000:{bus:02x}:00.0' for bus in range(8)],
                                       device_probe_observed=True, ports_checked=[], occupied_ports=[],
                                       KFD_clean=True, PID_clean=True, ports_clean=True,
                                       membership_valid=True, active_steps=[])
        membership = dict(ledger_SHA256=sha(TRAINING80_LEDGER), parents=parents,
                          distinct_nodes=80, allocated_DCU=640, membership_qualified=True)
        return dict(observed_epoch=time.time(), complete=True, total_cards=640,
                    membership=membership, nodes=nodes)

    def training80_ready(self, payload, zone, count, label, *, parents=None, ports=(), donors=()):
        binding = dict(zone=zone, nodes_required=count, publication_commit='a'*40,
                       allowed_nodes=[name for name, row in payload['nodes'].items()
                                      if row['zone'] == zone or row['zone'] in donors],
                       required_ports=list(ports))
        if parents is not None:
            binding['allowed_parents'] = parents
        if donors:
            binding.update(backfill_zones=list(donors), deadline_s=300)
        ident = digest([label, binding])
        self.q.db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?)',
                          (ident, self.rows[0]['id'], 'component', 'READY', canonical(binding).decode(),
                           '[]', None, None, time.time()))
        return ident

    def test_training80_can_plan_all_nodes_without_cross_parent_steps(self):
        payload = self.training80_payload()
        parents = payload['membership']['parents']
        for parent, observation in parents.items():
            for zone, nodes in observation['coverage_by_group_and_node'].items():
                self.training80_ready(payload, zone, len(nodes), parent, parents=[parent])
        fleet = receive_snapshot(payload, time.monotonic(), 'training80-fixture', 'a'*64)
        plan = self.q.plan(fleet)
        selected = [name for job in plan['jobs'] for name in job['nodes']]
        self.assertEqual(len(plan['jobs']), 32)
        self.assertEqual((len(selected), set(selected)), (80, set(payload['nodes'])))
        self.assertEqual(plan['selected_cards'], 640)
        self.assertEqual(plan['actual_submitted_cards'], 0)
        self.assertIsNone(plan['busy_percent'])
        self.assertEqual(plan['unplaced_ready_runs'], [])
        self.assertTrue(all(set(states) == {'planned'} for parents in plan['node_coverage'].values()
                            for states in parents.values()))
        for job in plan['jobs']:
            self.assertEqual({payload['nodes'][name]['parent'] for name in job['nodes']}, {job['parent']})
        self.q.reserve(plan, fleet)
        self.assertEqual(self.q.summary()['runs'], {'RESERVED': 32})

    def test_training80_explains_unplaced_nodes_and_ready_runs(self):
        payload = self.training80_payload()
        parent = min(payload['membership']['parents'])
        parents = payload['membership']['parents']
        own = parents[parent]['coverage_by_group_and_node']
        reserved, planned = sorted(own['T01'])
        active, unobserved = sorted(own['T02'])
        unclean, unmatched = sorted(own['T03'])
        held = self.training80_ready(payload, 'T01', 1, 'held', parents=[parent])
        self.q.db.execute('UPDATE runs SET state=?, nodes=? WHERE id=?',
                          ('UNKNOWN', json.dumps([reserved]), held))
        parents[parent]['active_steps'][parent + '.123'] = {'member_nodes': [active]}
        payload['nodes'][unobserved]['device_probe_observed'] = False
        payload['nodes'][unclean]['PID_clean'] = False
        self.training80_ready(payload, 'T01', 1, 'next', parents=[parent])
        waiting = self.training80_ready(payload, 'T03', 2, 'needs-two', parents=[parent])
        fleet = receive_snapshot(payload, time.monotonic(), 'training80-fixture', 'a'*64)

        plan = self.q.plan(fleet)
        coverage = plan['node_coverage']
        self.assertEqual(coverage['T01'][parent], {'queue_held': [reserved], 'planned': [planned]})
        self.assertEqual(coverage['T02'][parent], {'slurm_step_present_or_unverified': [active],
                                                   'admission_unverified': [unobserved]})
        self.assertEqual(coverage['T03'][parent], {'device_cleanliness_unconfirmed': [unclean],
                                                   'available_without_placement': [unmatched]})
        nodes = [node for parents_by_zone in coverage.values() for states in parents_by_zone.values()
                 for names in states.values() for node in names]
        self.assertCountEqual(nodes, payload['nodes'].keys())
        self.assertEqual(plan['unplaced_ready_runs'], [dict(run=waiting, zone='T03', nodes_required=2)])
        self.assertEqual(plan['selected_cards'], 8)
        self.assertEqual(plan['actual_submitted_cards'], 0)

    def test_training80_fills_both_donor_pools_without_displacing_local_work(self):
        payload = self.training80_payload()
        first, other = sorted(payload['membership']['parents'])
        for row in payload['nodes'].values():
            if row['parent'] == first and row['zone'] == 'T02':
                row['ports_checked'] = [44000]
        early = self.training80_ready(payload, 'O01', 1, 'early', parents=[first], ports=[44000], donors=['T02'])
        for parent, observation in payload['membership']['parents'].items():
            for zone, nodes in observation['coverage_by_group_and_node'].items():
                if zone != 'T02':
                    self.training80_ready(payload, zone, len(nodes), parent, parents=[parent])
        own = self.training80_ready(payload, 'T02', 1, 'own', parents=[first])
        self.training80_ready(payload, 'O01', 2, 'second-parent', parents=[other], donors=['T02'])
        fleet = receive_snapshot(payload, time.monotonic(), 'training80-backfill-fixture', 'a'*64)

        plan = self.q.plan(fleet)
        selected = [node for job in plan['jobs'] for node in job['nodes']]
        self.assertEqual((len(selected), set(selected)), (80, set(payload['nodes'])))
        self.assertEqual(plan['actual_submitted_cards'], 0)
        self.assertIsNone(plan['busy_percent'])
        loans = [job for job in plan['jobs'] if 'backfill' in job]
        self.assertEqual([(job['parent'], len(job['nodes'])) for job in loans], [(first, 1), (other, 2)])
        self.assertEqual(loans[0]['run'], early)
        own_job = next(job for job in plan['jobs'] if job['run'] == own)
        self.assertNotIn('backfill', own_job)
        self.assertEqual(own_job['parent'], first)

        self.q.reserve(plan, fleet)
        row = self.q.db.execute('SELECT data FROM events WHERE run=? AND kind="reserved"', (early,)).fetchone()
        self.assertEqual(json.loads(row['data'])['backfill'],
                         dict(donor_zone='T02', borrower_zone='O01', deadline_s=300,
                              return_condition='actual_terminal_and_post_clean'))
        self.q.submission(early, dict(request_id='request1', intent='original-intent', parent=first), unknown=True)
        waiting = self.training80_ready(payload, 'T02', 1, 'wait-for-return', parents=[first])
        self.assertEqual(self.q.plan(fleet)['jobs'], [])
        with self.assertRaisesRegex(ValueError, 'terminal and post'):
            self.q.finish(early, dict(self.result(), parent=first, post_clean=False))
        self.q.finish(early, dict(self.result(), parent=first))
        next_job, = self.q.plan(fleet)['jobs']
        self.assertEqual((next_job['run'], next_job['nodes']), (waiting, loans[0]['nodes']))

    def test_training80_backfill_respects_donor_assignment_ports_and_opt_in(self):
        payload = self.training80_payload()
        parent = '1087656'
        self.training80_ready(payload, 'I01', 4, 'owner', parents=[parent])
        self.training80_ready(payload, 'I01', 1, 'no-opt-in', parents=[parent])
        self.training80_ready(payload, 'I01', 2, 'partial-donor', parents=[parent], donors=['T02'])
        self.training80_ready(payload, 'I01', 1, 'unchecked-port', parents=[parent], ports=[44001], donors=['T02'])
        available = self.training80_ready(payload, 'I01', 1, 'verified-port', parents=[parent], ports=[44002],
                                          donors=['T02'])
        observation = payload['membership']['parents'][parent]
        active, clean = sorted(observation['coverage_by_group_and_node']['T02'])
        observation['active_steps'][parent + '.123'] = {'member_nodes': [active]}
        payload['nodes'][clean].update(ports_checked=[44002])
        fleet = receive_snapshot(payload, time.monotonic(), 'training80-backfill-fixture', 'a'*64)

        loans = [job for job in self.q.plan(fleet)['jobs'] if 'backfill' in job]
        self.assertEqual([(job['run'], job['nodes']) for job in loans], [(available, [clean])])

    def test_training80_refits_single_node_loans_across_donors(self):
        payload = self.training80_payload()
        first, other = sorted(payload['membership']['parents'])
        slots = [('T02', first, [44000, 44002]), ('T03', first, [44000, 44001]),
                 ('T04', other, [44001])]
        nodes = []
        for zone, parent, ports in slots:
            node = min(name for name, row in payload['nodes'].items()
                       if row['zone'] == zone and row['parent'] == parent)
            payload['nodes'][node]['ports_checked'] = ports
            nodes.append(node)

        flexible = self.training80_ready(payload, 'O01', 1, 'flexible', parents=[first],
                                         ports=[44000], donors=['T02', 'T03'])
        bridge = self.training80_ready(payload, 'O01', 1, 'bridge', parents=[first, other],
                                       ports=[44001], donors=['T03', 'T04'])
        fixed = self.training80_ready(payload, 'O01', 1, 'fixed', parents=[first],
                                      ports=[44002], donors=['T02'])
        fleet = receive_snapshot(payload, time.monotonic(), 'training80-backfill-fixture', 'a'*64)

        plan = self.q.plan(fleet)
        self.assertEqual([(job['run'], *job['nodes'], job['parent'], job['backfill']['donor_zone'])
                          for job in plan['jobs']],
                         [(flexible, nodes[1], first, 'T03'), (bridge, nodes[2], other, 'T04'),
                          (fixed, nodes[0], first, 'T02')])
        self.assertEqual(plan['selected_cards'], 24)
        self.assertEqual(plan['actual_submitted_cards'], 0)
        self.q.reserve(plan, fleet)
        self.assertEqual(self.q.reserved_parent(bridge), other)

    def test_training80_refits_two_node_loans_through_two_authorized_donors(self):
        payload = self.training80_payload()
        first, other = sorted(payload['membership']['parents'])
        for info in payload['nodes'].values():
            if (info['parent'], info['zone']) in ((other, 'T04'), (first, 'T02'), (first, 'T03')):
                info['ports_checked'] = [44001]
        flexible = self.training80_ready(payload, 'O01', 2, 'two-node-flex', parents=[first],
                                         ports=[44001], donors=['T02', 'T03'])
        bridge = self.training80_ready(payload, 'O01', 2, 'two-node-bridge', parents=[first, other],
                                       ports=[44001], donors=['T03', 'T04'])
        fixed = self.training80_ready(payload, 'O01', 2, 'two-node-fixed', parents=[first],
                                      ports=[44001], donors=['T02'])
        fleet = receive_snapshot(payload, time.monotonic(), 'training80-multi-refit', 'c'*64)

        plan = self.q.plan(fleet)
        self.assertEqual([(j['run'], j['backfill']['donor_zone'], j['parent']) for j in plan['jobs']],
                         [(flexible, 'T03', first), (bridge, 'T04', other), (fixed, 'T02', first)])
        self.assertEqual(len({node for job in plan['jobs'] for node in job['nodes']}), 6)
        self.assertEqual(plan['selected_cards'], 48)
        self.q.reserve(plan, fleet)
        self.assertEqual(self.q.reserved_parent(bridge), other)

    def test_training80_two_node_refit_never_moves_an_active_loan(self):
        payload = self.training80_payload()
        first = min(payload['membership']['parents'])
        for info in payload['nodes'].values():
            if info['parent'] == first and info['zone'] in ('T02', 'T03'):
                info['ports_checked'] = [44001]
        original = self.training80_ready(payload, 'O01', 2, 'first-loan', parents=[first],
                                         ports=[44001], donors=['T02', 'T03'])
        fleet = receive_snapshot(payload, time.monotonic(), 'training80-existing-loan', 'd'*64)
        plan = self.q.plan(fleet)
        self.q.reserve(plan, fleet)
        self.q.submission(original, dict(request_id='original', intent='original', parent=first), unknown=True)
        self.training80_ready(payload, 'O01', 2, 'blocked', parents=[first], ports=[44001], donors=['T02'])

        result = self.q.plan(fleet)
        self.assertEqual(result['jobs'], [])
        self.assertEqual(self.q.dispatch_state(original)['nodes'], plan['jobs'][0]['nodes'])

    def test_partial_refit_at_search_limit_keeps_the_valid_existing_plan(self):
        borrower = dict(zone='O01', nodes_required=2, deadline_s=300)
        first = dict(run='first', nodes=['a', 'b'], parent='1087656',
                     backfill=dict(donor_zone='T01', borrower_zone='O01', deadline_s=300,
                                   return_condition='actual_terminal_and_post_clean'))
        second = dict(run='second')
        parent = '1087656'
        free = {('T01', parent): ['a', 'b'], ('T02', parent): ['c', 'd']}
        waiting = [(first, borrower, {'a', 'b', 'c', 'd'}, [parent], ['T01', 'T02']),
                   (second, borrower, {'a', 'b'}, [parent], ['T01'])]
        original = copy.deepcopy(first)
        self.assertEqual(refit_provisional_loans(waiting, free, [first], maximum_choices=1), [])
        self.assertEqual(first, original)
        self.assertEqual(second, {'run': 'second'})

    def test_training80_backfill_rejects_unbound_donor_and_deadline(self):
        payload = self.training80_payload()
        ident = self.training80_ready(payload, 'I01', 1, 'loan', parents=['1087656'], donors=['T02'])
        base = json.loads(self.q.db.execute('SELECT binding FROM runs WHERE id=?', (ident,)).fetchone()[0])
        fleet = receive_snapshot(payload, time.monotonic(), 'training80-backfill-fixture', 'a'*64)
        for changes, reason in [({'backfill_zones': ['I01']}, 'invalid training80 backfill zones'),
                                ({'backfill_zones': ['T02', 'T02']}, 'invalid training80 backfill zones'),
                                ({'backfill_zones': ['T99']}, 'invalid training80 backfill zones'),
                                ({'deadline_s': 0}, 'bounded run deadline')]:
            with self.subTest(changes=changes):
                self.q.db.execute('UPDATE runs SET binding=? WHERE id=?',
                                  (canonical(dict(base, **changes)).decode(), ident))
                with self.assertRaisesRegex(ValueError, reason):
                    self.q.plan(fleet)

    def test_training80_keeps_occupied_nodes_and_unknown_reservations_out(self):
        payload = self.training80_payload()
        parents = payload['membership']['parents']
        first, other = sorted(parents)
        coverage = parents[other]['coverage_by_group_and_node']['I01']
        occupied = next(iter(coverage))
        coverage[occupied] = dict(state='PARTIALLY_ASSIGNED', numbered_step_assigned_DCU_bounds=[2, 2])
        self.training80_ready(payload, 'I01', 5, 'cross-parent', parents=list(parents))
        self.training80_ready(payload, 'I01', 1, 'unbound')
        smaller = self.training80_ready(payload, 'I01', 2, 'pack', parents=list(parents))
        fleet = receive_snapshot(payload, time.monotonic(), 'training80-fixture', 'a'*64)
        plan = self.q.plan(fleet)
        self.assertEqual([(job['run'], job['parent']) for job in plan['jobs']], [(smaller, other)])
        self.assertNotIn(occupied, plan['jobs'][0]['nodes'])
        self.q.reserve(plan, fleet)
        handle = dict(request_id='request1', intent='original-intent', parent=first)
        with self.assertRaisesRegex(ValueError, 'submission parent differs'):
            self.q.submission(smaller, handle, unknown=True)
        self.q.submission(smaller, dict(handle, parent=other), unknown=True)
        larger = self.training80_ready(payload, 'I01', 4, 'whole-group', parents=list(parents))
        plan = self.q.plan(fleet)
        self.assertEqual([(job['run'], job['parent']) for job in plan['jobs']], [(larger, first)])
        self.q.reserve(plan, fleet)
        self.training80_ready(payload, 'I01', 2, 'keep-unknown', parents=[other])
        self.assertEqual(self.q.plan(fleet)['jobs'], [])
        self.assertEqual(self.q.summary()['runs'], {'READY': 3, 'RESERVED': 1, 'UNKNOWN': 1})
        with self.assertRaisesRegex(ValueError, 'terminal parent differs'):
            self.q.finish(smaller, dict(self.result(), parent=first))
        self.assertEqual(self.q.plan(fleet)['jobs'], [])
        self.q.finish(smaller, dict(self.result(), parent=other))
        self.assertEqual([job['parent'] for job in self.q.plan(fleet)['jobs']], [other])

    def test_training80_rejects_incomplete_or_mismatched_placement(self):
        for fault, message in [('ledger', 'bind the active node ledger'), ('parent', 'parent or group drift'),
                               ('missing_node', 'device inventory differs'), ('owner', 'parent ownership'),
                               ('unlocated_step', 'step locations are unverified'),
                               ('foreign_node_step', 'step escaped its parent')]:
            with self.subTest(fault=fault):
                payload = self.training80_payload()
                name = next(iter(payload['nodes']))
                parent = payload['nodes'][name]['parent']
                observation = payload['membership']['parents'][parent]
                if fault == 'ledger':
                    payload['membership']['ledger_SHA256'] = '0'*64
                elif fault == 'parent':
                    payload['nodes'][name]['parent'] = next(key for key in payload['membership']['parents'] if key != parent)
                elif fault == 'missing_node':
                    del payload['nodes'][name]
                elif fault == 'owner':
                    observation['owned_by_observer'] = False
                elif fault == 'unlocated_step':
                    observation['step_locations_exact'] = False
                else:
                    observation['active_steps'][parent + '.123'] = {'member_nodes': ['foreign-node']}
                fleet = receive_snapshot(payload, time.monotonic(), 'training80-fixture', 'a'*64)
                with self.assertRaisesRegex(ValueError, message):
                    self.q.plan(fleet)

    def test_training80_port_conflict_does_not_block_unrelated_task(self):
        payload = self.training80_payload()
        nodes = payload['nodes']
        parent = '1087656'
        group = [node for node in nodes.values() if node['parent'] == parent and node['zone'] == 'T01']
        for node in group:
            node.update(ports_checked=[43000, 43001], occupied_ports=[43000], ports_clean=False)
        self.training80_ready(payload, 'T01', 2, 'occupied', parents=[parent], ports=[43000])
        self.training80_ready(payload, 'T01', 2, 'unverified', parents=[parent], ports=[43002])
        ready = self.training80_ready(payload, 'T01', 2, 'available', parents=[parent], ports=[43001])
        fleet = receive_snapshot(payload, time.monotonic(), 'training80-fixture', 'a'*64)
        self.assertEqual([job['run'] for job in self.q.plan(fleet)['jobs']], [ready])

    def test_training80_refits_flexible_run_to_other_parent(self):
        payload = self.training80_payload()
        first, other = sorted(payload['membership']['parents'])
        flexible = self.training80_ready(payload, 'T01', 2, 'flex', parents=[first, other])
        fixed = self.training80_ready(payload, 'T01', 2, 'fixed', parents=[first])
        fleet = receive_snapshot(payload, time.monotonic(), 'training80-fixture', 'a'*64)

        plan = self.q.plan(fleet)
        self.assertEqual([(job['run'], job['parent']) for job in plan['jobs']],
                         [(flexible, other), (fixed, first)])
        self.assertEqual(plan['selected_cards'], 32)
        self.q.reserve(plan, fleet)
        self.assertEqual(self.q.reserved_parent(flexible), other)
        self.assertEqual(self.q.reserved_parent(fixed), first)

    def test_training80_refits_for_port_and_preserves_fifo_when_a_run_cannot_fit(self):
        payload = self.training80_payload()
        parent = '1087656'
        zone = 'I01'
        nodes = sorted(name for name, row in payload['nodes'].items()
                       if row['parent'] == parent and row['zone'] == zone)
        payload['nodes'][nodes[0]]['ports_checked'] = [43000]
        flexible = self.training80_ready(payload, zone, 2, 'flex', parents=[parent])
        self.training80_ready(payload, zone, 3, 'too-big', parents=[parent])
        fixed = self.training80_ready(payload, zone, 1, 'port', parents=[parent], ports=[43000])
        tail = self.training80_ready(payload, zone, 1, 'tail', parents=[parent])
        fleet = receive_snapshot(payload, time.monotonic(), 'training80-fixture', 'a'*64)

        plan = self.q.plan(fleet)
        self.assertEqual([job['run'] for job in plan['jobs']], [flexible, fixed, tail])
        self.assertEqual(plan['jobs'][1]['nodes'], [nodes[0]])
        self.assertEqual(sorted(node for job in plan['jobs'] for node in job['nodes']), nodes)

    def test_concurrent_reservation_one_winner(self):
        self.ready();plan=self.q.plan(self.fleet);results=[]
        def reserve():
            q=Queue(self.path)
            try:q.reserve(plan,self.fleet);results.append(True)
            except ValueError:results.append(False)
            finally:q.db.close()
        threads=[threading.Thread(target=reserve)for _ in range(2)]
        for t in threads:t.start()
        for t in threads:t.join()
        self.assertEqual(sorted(results),[False,True])

    def claimable_reservation(self):
        ident = self.ready()
        plan = self.q.plan(self.fleet)
        ticket = self.q.reserve(plan, self.fleet, claimable=True)[ident]
        handle = dict(request_id='request1', intent='original-intent', nodes=plan['jobs'][0]['nodes'])
        return ident, ticket, handle

    def test_released_ticket_cannot_claim_new_reservation(self):
        ident, ticket, handle = self.claimable_reservation()
        self.assertEqual(self.q.dispatch_state(ident)['reservation']['dispatch_ticket'], ticket)
        with self.assertRaisesRegex(ValueError, 'claim dispatch'):
            self.q.submission(ident, handle)
        self.q.release_unclaimed(ident, ticket)
        self.assertEqual(self.q.dispatch_state(ident), dict(state='READY', nodes=[], handle=None, reservation=None))
        payload = {key: value for key, value in self.fleet.items() if key != 'local_receipt'}
        fresh = receive_snapshot(payload, time.monotonic(), 'second-fixture', 'b'*64)
        new_ticket = self.q.reserve(self.q.plan(fresh), fresh, claimable=True)[ident]
        self.assertNotEqual(ticket, new_ticket)
        with self.assertRaisesRegex(ValueError, 'ticket required'):
            self.q.claim_dispatch(ident, ticket, handle)
        self.q.claim_dispatch(ident, new_ticket, handle)
        self.assertEqual(self.q.dispatch_state(ident)['handle'], handle)

    def test_claimed_nodes_stay_reserved_after_lost_launcher(self):
        ident, ticket, handle = self.claimable_reservation()
        with self.assertRaisesRegex(ValueError, 'dispatch nodes'):
            self.q.claim_dispatch(ident, ticket, dict(handle, nodes=['elsewhere']))
        self.q.claim_dispatch(ident, ticket, handle)
        self.q.db.close()
        self.q = Queue(self.path)
        self.addCleanup(self.q.db.close)
        self.assertEqual(self.q.dispatch_state(ident)['state'], 'DISPATCHING')
        with self.assertRaisesRegex(ValueError, 'unclaimed reservation'):
            self.q.release_unclaimed(ident, ticket)
        with self.assertRaisesRegex(ValueError, 'different execution identity'):
            self.q.submission(ident, dict(handle, request_id='retry'))
        waiting = self.ready('waiting', count=4)
        self.assertEqual(self.q.plan(self.fleet)['jobs'], [])
        with self.assertRaisesRegex(ValueError, 'post'):
            self.q.finish(ident, dict(self.result(), post_clean=False))
        self.q.finish(ident, self.result())
        self.assertEqual([job['run'] for job in self.q.plan(self.fleet)['jobs']], [waiting])

    def test_claim_and_release_one_winner(self):
        ident, ticket, handle = self.claimable_reservation()
        barrier = threading.Barrier(2)
        winners = []

        def compete(action, state, *args):
            queue = Queue(self.path)
            try:
                barrier.wait(timeout=10)
                getattr(queue, action)(ident, ticket, *args)
                winners.append(state)
            except ValueError:
                pass
            finally:
                queue.db.close()

        claim = threading.Thread(target=compete, args=('claim_dispatch', 'DISPATCHING', handle))
        release = threading.Thread(target=compete, args=('release_unclaimed', 'READY'))
        for thread in claim, release:
            thread.start()
        for thread in claim, release:
            thread.join(timeout=15)
            self.assertFalse(thread.is_alive())
        self.assertEqual(winners, [self.q.dispatch_state(ident)['state']])

    def test_claimed_submission_keeps_original_identity_and_nodes(self):
        ident, ticket, handle = self.claimable_reservation()
        self.q.claim_dispatch(ident, ticket, handle)
        with self.assertRaisesRegex(ValueError, 'submission nodes'):
            self.q.submission(ident, dict(handle, nodes=['elsewhere']))
        actual = dict(request_id=handle['request_id'], intent=handle['intent'], transport_request_id='transport-1')
        self.q.submission(ident, actual, unknown=True)
        state = self.q.dispatch_state(ident)
        self.assertEqual(state['state'], 'UNKNOWN')
        self.assertEqual(state['handle'], {**handle, **actual})

    def test_legacy_reservation_requires_original_resolution(self):
        ident = self.ready()
        self.q.reserve(self.q.plan(self.fleet), self.fleet)
        with self.assertRaisesRegex(ValueError, 'unclaimed reservation'):
            self.q.release_unclaimed(ident, 'proposed-ticket')
        self.assertEqual(self.q.dispatch_state(ident)['state'], 'RESERVED')

    def test_expired_ticket_can_release_but_cannot_start(self):
        ident, ticket, handle = self.claimable_reservation()
        later, wall = time.monotonic() + 61, time.time() + 61
        with patch('tuning_queue.time.monotonic', return_value=later), patch('tuning_queue.time.time', return_value=wall):
            with self.assertRaisesRegex(ValueError, 'admission expired'):
                self.q.claim_dispatch(ident, ticket, handle)
            self.q.release_unclaimed(ident, ticket)
        self.assertEqual(self.q.dispatch_state(ident)['state'], 'READY')

    def test_training80_replans_expired_unclaimed_nodes_with_new_ticket(self):
        payload = self.training80_payload()
        parent = '1087656'
        ident = self.training80_ready(payload, 'T01', 2, 'expired', parents=[parent])
        original_snapshot = receive_snapshot(payload, time.monotonic(), 'training80-first', 'a'*64)
        prior_plan = self.q.plan(original_snapshot)
        old_ticket = self.q.reserve(prior_plan, original_snapshot, claimable=True)[ident]
        handle = dict(request_id='original', intent='original', parent=parent, nodes=prior_plan['jobs'][0]['nodes'])
        self.assertEqual(self.q.plan(original_snapshot)['jobs'], [])

        later, wall = time.monotonic() + 61, time.time() + 61
        with patch('tuning_queue.time.monotonic', return_value=later), patch('tuning_queue.time.time', return_value=wall):
            with self.assertRaisesRegex(ValueError, 'fresh'):
                self.q.plan(original_snapshot)
            fresh = receive_snapshot(payload, later, 'training80-second', 'b'*64)
            plan = self.q.plan(fresh)
            self.assertEqual(list(plan['expired_unclaimed_reservations']), [ident])
            self.assertEqual(plan['jobs'], prior_plan['jobs'])
            self.assertEqual(self.q.dispatch_state(ident)['state'], 'RESERVED')
            with self.assertRaisesRegex(ValueError, 'new dispatch ticket'):
                self.q.reserve(plan, fresh)
            self.assertEqual(self.q.dispatch_state(ident)['reservation']['dispatch_ticket'], old_ticket)
            new_ticket = self.q.reserve(plan, fresh, claimable=True)[ident]
            self.assertNotEqual(old_ticket, new_ticket)
            with self.assertRaisesRegex(ValueError, 'ticket required'):
                self.q.claim_dispatch(ident, old_ticket, handle)
            self.q.claim_dispatch(ident, new_ticket, dict(handle, request_id='fresh'))
            self.assertEqual(self.q.dispatch_state(ident)['state'], 'DISPATCHING')
        self.assertEqual([row[0] for row in self.q.db.execute('SELECT kind FROM events WHERE run=? ORDER BY seq', (ident,))],
                         ['reserved', 'expired_unclaimed', 'reserved', 'dispatch_claimed'])

    def test_expiry_reconciliation_preserves_claimed_unknown_and_legacy_nodes(self):
        legacy = self.ready('legacy', zone='Kimi', count=4)
        self.q.reserve(self.q.plan(self.fleet), self.fleet)
        claimed, claimed_ticket, handle = self.claimable_reservation()
        self.q.claim_dispatch(claimed, claimed_ticket, handle)
        unknown = self.ready('unknown', zone='DeepSeek', count=8)
        expired = self.ready('expired', zone='GLM', count=2)
        prior_plan = self.q.plan(self.fleet)
        tickets = self.q.reserve(prior_plan, self.fleet, claimable=True)
        unknown_job = next(job for job in prior_plan['jobs'] if job['run'] == unknown)
        unknown_handle = dict(request_id='unknown', intent='unknown', nodes=unknown_job['nodes'])
        self.q.claim_dispatch(unknown, tickets[unknown], unknown_handle)
        self.q.submission(unknown, unknown_handle, unknown=True)
        later, wall = time.monotonic() + 61, time.time() + 61
        with patch('tuning_queue.time.monotonic', return_value=later), patch('tuning_queue.time.time', return_value=wall):
            payload = {key: value for key, value in self.fleet.items() if key != 'local_receipt'}
            fresh = receive_snapshot(payload, later, 'after-expiry', 'b'*64)
            plan = self.q.plan(fresh)
            self.assertEqual(list(plan['expired_unclaimed_reservations']), [expired])
            self.q.reserve(plan, fresh, claimable=True)
            self.assertEqual(self.q.dispatch_state(claimed)['state'], 'DISPATCHING')
            self.assertEqual(self.q.dispatch_state(unknown)['state'], 'UNKNOWN')
            self.assertEqual(self.q.dispatch_state(legacy)['state'], 'RESERVED')
            self.assertIsNone(self.q.dispatch_state(legacy)['reservation'].get('dispatch_ticket'))

    def test_plan_with_expiry_rejects_concurrent_reservation_change(self):
        ident, ticket, _ = self.claimable_reservation()
        later, wall = time.monotonic() + 61, time.time() + 61
        with patch('tuning_queue.time.monotonic', return_value=later), patch('tuning_queue.time.time', return_value=wall):
            payload = {key: value for key, value in self.fleet.items() if key != 'local_receipt'}
            fresh = receive_snapshot(payload, later, 'after-expiry', 'b'*64)
            plan = self.q.plan(fresh)
            other = Queue(self.path)
            self.addCleanup(other.db.close)
            other.release_unclaimed(ident, ticket)
            new_ticket = other.reserve(other.plan(fresh), fresh, claimable=True)[ident]
            with self.assertRaisesRegex(ValueError, 'plan changed'):
                self.q.reserve(plan, fresh, claimable=True)
            self.assertEqual(self.q.dispatch_state(ident)['reservation']['dispatch_ticket'], new_ticket)

    def test_retire_prelaunch_blocks_dispatch_without_fake_terminal(self):
        ident=self.ready();self.q.retire_ready(ident,'immutable source timeout mismatch','e'*64)
        self.assertEqual(self.q.plan(self.fleet)['jobs'],[])
        row=self.q.db.execute('SELECT state,result FROM runs WHERE id=?',(ident,)).fetchone()
        self.assertEqual(row['state'],'RETIRED_PRELAUNCH');self.assertFalse(json.loads(row['result'])['measurement_terminal'])
        self.assertFalse(json.loads(row['result'])['GPU_submission'])

    def test_retire_cannot_free_reserved_or_unknown_work(self):
        ident=self.ready();plan=self.q.plan(self.fleet);self.q.reserve(plan,self.fleet)
        with self.assertRaisesRegex(ValueError,'only unreserved'):self.q.retire_ready(ident,'defect','e'*64)
        self.q.submission(ident,dict(request_id='original',intent='original'),True)
        with self.assertRaisesRegex(ValueError,'only unreserved'):self.q.retire_ready(ident,'defect','e'*64)
        self.assertEqual(self.q.summary()['runs']['UNKNOWN'],1)

    def test_retire_requires_reason_and_exact_evidence(self):
        ident=self.ready()
        with self.assertRaises(ValueError):self.q.retire_ready(ident,'','e'*64)
        with self.assertRaises(ValueError):self.q.retire_ready(ident,'defect','short')
        self.assertEqual(self.q.summary()['runs']['READY'],1)

    def test_unknown_not_replayed(self):
        ident=self.active(True)
        with self.assertRaisesRegex(ValueError,'no replay'):
            self.q.submission(ident,dict(request_id='request2',intent='new'))

    def test_wrong_identity_cannot_release(self):
        ident=self.active();r=self.result();r['request_id']='different'
        with self.assertRaisesRegex(ValueError,'identity'):self.q.finish(ident,r)
        self.assertEqual(self.q.summary()['runs']['SUBMITTED'],1)

    def test_missing_post_cannot_release(self):
        ident=self.active();r=self.result();r['post_clean']=False
        with self.assertRaisesRegex(ValueError,'post'):self.q.finish(ident,r)

    def test_failure_preserved_can_release_only_after_post(self):
        ident=self.active(True);r=self.result();self.q.finish(ident,r)
        saved=json.loads(self.q.db.execute('SELECT result FROM runs WHERE id=?',(ident,)).fetchone()[0])
        self.assertFalse(saved['keeper_passed']);self.assertFalse(saved['passed']);self.assertEqual(saved['planned'],20)
        self.assertEqual(self.q.summary()['runs']['TERMINAL'],1)

    def test_denominator_and_keeper_failure_prevent_promotion(self):
        ident=self.active();r=self.result();r['missing']=0
        with self.assertRaisesRegex(ValueError,'denominator'):self.q.finish(ident,r)
        r=self.result();r.update(promote=True)
        with self.assertRaisesRegex(ValueError,'cannot promote'):self.q.finish(ident,r)

    def test_failed_request_cannot_promote_but_terminal_can_release(self):
        ident = self.active()
        result = self.result()
        result.update(passed=True, keeper_passed=True, promote=True, completed=19,
                      failed=1, rejected=0, canceled=0, missing=0)
        with self.assertRaisesRegex(ValueError, "cannot promote"):
            self.q.finish(ident, result)
        self.assertEqual(self.q.summary()["runs"], {"SUBMITTED": 1})

        result.update(passed=False, promote=False)
        self.q.finish(ident, result)
        self.assertEqual(self.q.summary()["runs"], {"TERMINAL": 1})
        saved = self.q.db.execute("SELECT result FROM runs WHERE id=?", (ident,)).fetchone()[0]
        self.assertEqual(json.loads(saved), result)

    def test_prepare_rejects_unpublished_before_launch(self):
        spec=self.rows[0]
        binding=dict(root_reviewed=True,candidate=spec['id'],stage='component',rung=0,workload_SHA256=digest(spec['workload']),memory_floor_bytes=spec['memory_floor_bytes'],deadline_s=600,nodes_required=spec['nodes_required'],zone=spec['zone'],prerequisites_closed=spec['prerequisites'],prerequisite_evidence=['exact-prior-receipt'],artifacts=[dict(role=k,path='code/bw1000/example/'+k,SHA256='b'*64)for k in ['source','deps','claim','launcher']],publication_commit='a'*40)
        with patch('tuning_queue.subprocess.check_output',side_effect=RuntimeError('not published')):
            with self.assertRaisesRegex(RuntimeError,'not published'):self.q.prepare(binding,ROOT)
        self.assertEqual(self.q.summary()['runs'],{})

    def test_component_cannot_be_ranked_with_model(self):
        with self.assertRaisesRegex(ValueError,'different'):
            select_next_rung([dict(comparison=self.comparison('component')),dict(comparison=self.comparison('model'))])

    def test_only_warm_resolved_winner_promotes(self):
        rows=[]
        for name,ratios in [('faster',[1.2]*9),('noise',[.9,1,1.1]*3),('slower',[.5]*9)]:
            rows.append(dict(candidate=name,comparison=self.comparison('service'),numeric_passed=True,post_clean=True,keeper_passed=True,planned=20,completed=20,rejected=0,canceled=0,missing=0,failed=0,baseline_completed=20,baseline_rejected=0,baseline_canceled=0,quality_correct=18,baseline_quality_correct=18,quality_screen_passed=True,warm_per_user_ratios=ratios))
        selected=select_next_rung(rows, min_pairs=9)
        self.assertEqual([r['candidate']for r in selected['promoted']],['faster'])
        self.assertFalse(selected['goal_qualified']);self.assertFalse(selected['formal_NI_proven'])

    def test_retention_is_at_most_one_third_of_unique_candidates(self):
        def winning_row(index):
            return dict(candidate=f"fast-{index}", comparison=self.comparison("service"),
                        numeric_passed=True, post_clean=True, keeper_passed=True,
                        planned=20, completed=20, rejected=0, canceled=0, missing=0, failed=0,
                        baseline_completed=20, baseline_rejected=0, baseline_canceled=0,
                        quality_correct=18, baseline_quality_correct=18, quality_screen_passed=True,
                        warm_per_user_ratios=[1.2] * 3)

        for size in range(7):
            with self.subTest(size=size):
                rows = [winning_row(i) for i in range(size)]
                ranked = select_next_rung(rows)
                self.assertEqual(len(ranked["promoted"]), size // 3)
                self.assertEqual(len(ranked["not_promoted"]), size - size // 3)

        with self.assertRaisesRegex(ValueError, "unique"):
            select_next_rung([winning_row(0), winning_row(0), winning_row(1)])
        with self.assertRaisesRegex(ValueError, "retention"):
            select_next_rung([winning_row(0)], keep_fraction=0.5)

    def test_rank_uses_exact_pair_budget_and_preserves_rejections(self):
        def winning_row(index, pairs):
            return dict(candidate=f"fast-{index}", comparison=self.comparison("service"),
                        numeric_passed=True, post_clean=True, keeper_passed=True,
                        planned=20, completed=20, rejected=0, canceled=0, missing=0, failed=0,
                        baseline_completed=20, baseline_rejected=0, baseline_canceled=0,
                        quality_correct=18, baseline_quality_correct=18, quality_screen_passed=True,
                        warm_per_user_ratios=[1.2 + index / 10] * pairs)

        for budget in (3, 9, 27):
            with self.subTest(budget=budget):
                ranked = select_next_rung([winning_row(0, budget - 1), winning_row(1, budget),
                                           winning_row(2, budget + 1)], min_pairs=budget)
                self.assertEqual([r["candidate"] for r in ranked["promoted"]], ["fast-1"])
                self.assertEqual({r["candidate"] for r in ranked["not_promoted"]}, {"fast-0", "fast-2"})
                self.assertTrue(all("exact" in r["reasons"][0] for r in ranked["not_promoted"]))

        with self.assertRaisesRegex(ValueError, "registered pair budget"):
            select_next_rung([winning_row(0, 4)], min_pairs=4)

    def test_fast_completed_subset_with_many_rejects_cannot_promote(self):
        row=dict(candidate='rejects-fast',comparison=self.comparison('service'),numeric_passed=True,post_clean=True,keeper_passed=True,planned=20,completed=5,rejected=15,canceled=0,missing=0,failed=0,baseline_completed=20,baseline_rejected=0,baseline_canceled=0,quality_correct=5,baseline_quality_correct=18,quality_screen_passed=True,warm_per_user_ratios=[3.0]*9)
        self.assertEqual(select_next_rung([row], min_pairs=9)['promoted'],[])

    def test_remote_wall_clock_skew_does_not_authorize_or_block(self):
        self.ready();self.fleet['observed_epoch']+=3600
        self.fleet['local_receipt']['payload_SHA256']=digest({k:v for k,v in self.fleet.items()if k!='local_receipt'})
        self.assertEqual(len(self.q.plan(self.fleet)['jobs']),1)
        self.fleet['local_receipt']['received_monotonic']-=61
        with self.assertRaisesRegex(ValueError,'fresh'):self.q.plan(self.fleet)

    def test_snapshot_cannot_be_restamped_or_survive_boot_change(self):
        with self.assertRaisesRegex(ValueError,'restamp'):receive_snapshot(self.fleet,time.monotonic(),'r','s')
        self.fleet['local_receipt']['boot_epoch_anchor']-=10
        with self.assertRaisesRegex(ValueError,'reboot'):self.q.plan(self.fleet)

    def test_kimi_static_baseline_geometry(self):
        p=ROOT/'code/bw1000/systematic-tuning-v1/kimi_tiling_cost.py'
        spec=importlib.util.spec_from_file_location('cost',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        # Longcase Q512/K3072/Q32:16querytiles each11FAcalls.
        self.assertEqual(m.count(512,32255,32,3072)['FA_calls'],176)
        self.assertEqual(m.count(1,32767,32,3072)['FA_calls'],11)
        self.assertEqual(len(m.study()['candidates']),12)
        with self.assertRaises(ValueError):m.cost(64,3072)

if __name__=='__main__':unittest.main(verbosity=2)
