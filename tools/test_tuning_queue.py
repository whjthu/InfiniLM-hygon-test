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

from tuning_queue import Queue,canonical,digest,expand,select_next_rung,receive_snapshot
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
        selected=select_next_rung(rows)
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

    def test_fast_completed_subset_with_many_rejects_cannot_promote(self):
        row=dict(candidate='rejects-fast',comparison=self.comparison('service'),numeric_passed=True,post_clean=True,keeper_passed=True,planned=20,completed=5,rejected=15,canceled=0,missing=0,failed=0,baseline_completed=20,baseline_rejected=0,baseline_canceled=0,quality_correct=5,baseline_quality_correct=18,quality_screen_passed=True,warm_per_user_ratios=[3.0]*9)
        self.assertEqual(select_next_rung([row])['promoted'],[])

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
