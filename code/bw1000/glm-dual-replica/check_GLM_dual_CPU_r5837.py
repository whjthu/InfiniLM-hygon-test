"""CPU timing/sequence/omission/EOS/termination tests; no Torch/native dlopen."""
import ast
import concurrent.futures
import hashlib
import json
import pathlib
import tempfile
import time

from glm_dual_clock_r5837 import ReceiptLedger, DualClock, ReceiptHub, PLAN
from kimi_cohort_feedback_idle_r4915 import FeedbackClient

checks=[]


def ok(name, value=True):
    assert value, name
    checks.append(name)


def reject(name, callback):
    try: callback()
    except (AssertionError, RuntimeError, ValueError): ok(name); return
    raise AssertionError('accepted invalid: '+name)


ledger=ReceiptLedger((0,1), 10., {99})
for replica in (0,1):
    for cohort in (0,1):
        rows=list(range(8*cohort,8*cohort+8))
        ledger.record(replica,cohort,0,rows,[11]*8,11.+replica+cohort/10,True)
# Last first decision is12.1. One early replica packet before this global
# anchor must be excluded; fill after-first times in receipt order instead.
ok('all32request_firsts_seen', all(len(h)==1 for rows in ledger.rows.values()for h in rows))
for replica in (0,1):
    for cohort in (0,1):
        rows=list(range(8*cohort,8*cohort+8))
        ledger.record(replica,cohort,1,rows,[99]*8,13.+replica+cohort/10)
q=ledger.finish()
ok('global_anchor_is_max_of32firsts', q['global_common_anchor_seconds']==2.0999999999999996)
ok('EOS_never_useful',q['decode_useful_tokens_strictly_after_anchor']==0)
ok('all32EOS_terminal_not_censored',q['completed']and not q['censored_rows'])
ok('cards32_even_isolated',q['all_provisioned_service_cards']==32)

ledger=ReceiptLedger((0,1), 0., {99})
for cohort in (0,1):ledger.record(0,cohort,0,list(range(8*cohort,8*cohort+8)),[11]*8,1.+cohort*.1,True)
# A emits useful decode before B's last first decision: those eight tokens
# must not enter the shared global decode numerator.
ledger.record(0,0,1,list(range(8)),[12]*8,2.)
for cohort in (0,1):ledger.record(1,cohort,0,list(range(8*cohort,8*cohort+8)),[21]*8,3.+cohort*.1,True)
ledger.record(0,1,1,list(range(8,16)),[12]*8,4.)
ledger.record(0,0,2,list(range(8)),[99]*8,4.1)
ledger.record(0,1,2,list(range(8,16)),[99]*8,4.2)
for cohort in (0,1):ledger.record(1,cohort,1,list(range(8*cohort,8*cohort+8)),[99]*8,5.+cohort*.1)
q=ledger.finish()
ok('strict_window_excludes_early_useful',q['decode_useful_tokens_strictly_after_anchor']==8)
ok('duration_last_decision_minus_last_first',abs(q['decode_window_seconds']-2.)<1e-12)
ok('rate_per_total32',abs(q['global_decode_tokens_s']-4.)<1e-12 and abs(q['decode_tokens_s_per_provisioned_card']-.125)<1e-12)
ok('TTFT_per_request_kept',len(q['per_request_collector_metrics'])==32)
ok('no_padding',q['padding_tokens']==0)
before=json.dumps(ledger.rows,sort_keys=True)
reject('no_afterEOS',lambda:ledger.record(0,0,3,list(range(8)),[12]*8,6.))
ok('invalid_packet_transactional',json.dumps(ledger.rows,sort_keys=True)==before)

missing=ReceiptLedger((0,),0.,{99})
missing.record(0,0,0,list(range(8)),[11]*8,1.,True)
q=missing.finish()
ok('partial_receipt_ids_retained',q['receipt_rows'][0][0][0]['id']==11)
ok('missing_first_and_unfinished_retained',not q['completed']and len(q['censored_rows'])==16)
ok('no_partial_throughput',q['global_decode_tokens_s']is None)
reject('row_omission',lambda:missing.record(0,1,0,list(range(8,15)),[11]*7,2.,True))
reject('duplicate_first',lambda:missing.record(0,0,0,list(range(8)),[11]*8,2.,True))
reject('skipped_sequence',lambda:missing.record(0,0,2,list(range(8)),[11]*8,2.))
reject('out_of_order_rows',lambda:missing.record(0,0,1,list(reversed(range(8))),[11]*8,2.))
reject('clock_backwards',lambda:missing.record(0,0,1,list(range(8)),[11]*8,.5))
reject('noninteger_token',lambda:missing.record(0,0,1,list(range(8)),[1.2]*8,2.))
reject('inactive_replica',lambda:missing.record(1,0,0,list(range(8)),[11]*8,2.,True))

with tempfile.TemporaryDirectory(prefix='glm-dual-CPU-')as temp:
    clock=DualClock(pathlib.Path(temp)/'receipts.json',{99},timeout=5)
    hubs=[ReceiptHub(i,clock,'127.0.0.1',0,'cpu-test-'+str(i),participants=2,emitter=1,timeout=5,idle_timeout=30)for i in(0,1)]
    clients=[]
    try:
        for hub in hubs:
            clients.append([FeedbackClient('127.0.0.1',hub.address[1],hub.nonce,rank,timeout=10)for rank in(0,1)])
        def replica_run(replica):
            receiver,emitter=clients[replica]
            for label,active in PLAN:
                begin=emitter.request('dual_begin',label)
                assert begin['active']==(replica in active)
                if replica in active:
                    emitter.request('begin',label,cohorts=[list(range(8)),list(range(8,16))])
                    for cohort in(0,1):
                        rows=list(range(8*cohort,8*cohort+8))
                        emitter.request('observe_first',label,cohort=cohort,rows=rows,ids=[11]*8)
                    for cohort in(0,1):
                        rows=list(range(8*cohort,8*cohort+8))
                        packet={'rows':rows,'ids':[99]*8,'stamp':.1,'seconds':.01}
                        emitter.request('publish',label,cohort=cohort,step=0,packet=packet)
                        assert receiver.request('take',label,cohort=cohort,step=0)==packet
                    emitter.request('end',label)
                result=emitter.request('dual_done',label,quality=True,
                    generated=[[11,99]for _ in range(16)]if replica in active else[],
                    stopped=[True]*16 if replica in active else[])
                assert result['qualified']
            emitter.request('dual_close','closed')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
            futures=[pool.submit(replica_run,i)for i in(0,1)]
            for f in futures:f.result(timeout=20)
        q=json.loads((pathlib.Path(temp)/'receipts.json').read_text())
        ok('two_existing_hubs_socket_protocol_sixphases',q['completed']and len(q['completed_phases'])==6)
        ok('both_replica_normalclose',q['closed_replicas']==[0,1]and q['failure']is None)
        ok('both_concurrent32_and_isolated16', [c['effective_requests']for c in q['completed_phases']]==[16,16,32,32,16,16])
        ok('packet_owner_fullIDs_join',all(c['quality_passed']for c in q['completed_phases']))
    finally:
        for group in clients:
            for client in group:client.close()
        for hub in hubs:hub.close()

with tempfile.TemporaryDirectory(prefix='glm-dual-preforward-CPU-')as temp:
    clock=DualClock(pathlib.Path(temp)/'reject.json',{99},timeout=5)
    def rejected_replica(replica):
        clock.begin(replica,PLAN[0][0])
        result=clock.finish(replica,PLAN[0][0],replica==1,[],[])
        assert not result['qualified']and not result['completed']
        clock.close_replica(replica)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
        fs=[pool.submit(rejected_replica,i)for i in(0,1)]
        for f in fs:f.result(timeout=10)
    q=json.loads((pathlib.Path(temp)/'reject.json').read_text())
    ok('preforward_rejection_both_close_noTPS',q['closed_replicas']==[0,1]and not q['completed']and q['normal_quality_rejection']and q['completed_phases'][0]['global_decode_tokens_s']is None)
    ok('preforward16censored_explicit',len(q['completed_phases'][0]['censored_rows'])==16)

# Only the declared immutable replica identity differs between the two main
# sources; they have the same native/math/loader statements.
a=pathlib.Path('modern_glm_dual_A_r5837.py').read_text()
b=pathlib.Path('modern_glm_dual_B_r5837.py').read_text()
ok('two_runner_only_replica_constant_diff',a.replace('REPLICA=0','REPLICA=1',1)==b)
probe=pathlib.Path('glm_dual_probe_r5837.py').read_text()
for name in ['glm_dual_clock_r5837.py','glm_dual_probe_r5837.py','modern_glm_dual_A_r5837.py','modern_glm_dual_B_r5837.py','GLM_transient_preflight_r5837.py','GLM_transient_preengine_r5837.py']:
    ast.parse(pathlib.Path(name).read_text());ok('AST:'+name)
ok('all_first_decisions_observed',"if step==0:feedback.request('observe_first'"in probe)
ok('original_complete_reference_gate_retained',"row['cross_geometry_full_IDs_passed'] = cross_vote.item() == CARDS"in probe)
ok('new_first_B_reference_not_B32',"if replica==0 else None"in probe)
ok('normalclose_fence_after_dual_close',probe.index("dual_request('dual_close'")<probe.index('closing=torch.ones'))
ok('original_native_image_unchanged','1af0b95df9db1c68f9e8c00124dbb0b331cd1b3650afd4506e3befa30f98be9d'in a)

out=pathlib.Path('GLM-dual-CPU-r5837.json');assert not out.exists()
names=['glm_dual_clock_r5837.py','glm_dual_probe_r5837.py','modern_glm_dual_A_r5837.py','modern_glm_dual_B_r5837.py','GLM_transient_preflight_r5837.py','GLM_transient_preengine_r5837.py','glm_dual_Model_r5837_inside.sh']
result={'completed':True,'checks':len(checks),'passed':checks,'scope':'Actual CPU ledger and existing socket-hub protocol. Not GPU/Model/throughput evidence.','source_SHA256':{n:hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()for n in names},'GPU_initialized':False,'native_compiled':False}
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'completed':True,'checks':len(checks),'output':str(out)}))
