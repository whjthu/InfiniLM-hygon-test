"""Offline original-byte audit of aborted zero-request residency and cleanup."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,tarfile
GPU=['910376.873','910374.1387','910371.781','910371.783','910380.580','910380.582','910376.875','910374.1389']
CPU=['910407.'+str(i)for i in range(928,936)]
NATIVE='9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'

def audit(path):
    selected={};owners={};count=0
    with tarfile.open(path,'r:gz')as tf:
        members=tf.getmembers();assert len({m.name for m in members})==len(members)
        manifest=json.load(tf.extractfile('MANIFEST.json'))
        assert len(manifest['files'])+1==len(members)
        for m in members:
            assert m.isfile()and not PurePosixPath(m.name).is_absolute()and'..'not in PurePosixPath(m.name).parts
            if m.name=='MANIFEST.json':continue
            raw=tf.extractfile(m).read();ref=manifest['files'][m.name]
            assert len(raw)==m.size==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['SHA256'];count+=1
            if m.name.startswith('slots/')and'/run/owner-'in m.name:
                q=json.loads(raw);parts=m.name.split('/');slot=int(parts[1]);stage,rank=map(int,parts[-1][6:-5].split('-'))
                assert q['completed']and q['worker_shutdown_completed']and q['normal_shutdown']and q['native_loop_drained']and q['post_close_Torch_liveness']
                assert q['Native_SHA256']==q['library_sha256']==NATIVE and q['attempted_Native']==q['completed_Native']==0 and q['service_requests']==[]
                assert q['actual_step']==GPU[slot]and q['run']=='fewshot-20261104-a-s'+str(slot)
                assert q['all16_preweight_8GiB_passed']and q['all16_loaded_and_postload_4GiB_passed']and q['constructor_noV_verified']
                owners[(slot,stage,rank)]=dict(normal=True,zero_requests=True,source_SHA256=ref['SHA256'])
            elif m.name.endswith('.json')and'/log-tail/'not in m.name:selected[m.name]=json.loads(raw)
    ledger=selected['cleanup/coordinator.json'];post=selected['cleanup/independent-post.json']
    assert ledger['completed']and ledger['STOP_committed']and not ledger.get('error')and ledger['GPU_submitted']==0
    assert not any('scancel'in s.get('argv',[])for s in ledger['signals']),'GPU forcedsignal unexpectedly used'
    assert len([s for s in ledger['signals']if s.get('signal')=='STOP'])==len([s for s in ledger['signals']if s.get('signal')=='KILL'])==8
    assert len(owners)==128 and set(owners)=={(s,t,r)for s in range(8)for t in range(2)for r in range(8)}
    late=[]
    for slot in range(8):
        base='slots/%d/'%slot;w=selected[base+'watch.json'];identity=selected[base+'run/FRONTEND.json']['identity']
        assert w['actual_GPU']==GPU[slot]and w['CPU_keeper']==CPU[slot]and not w['actual_client']
        assert identity==ledger['slots'][slot]['identity']
        assert selected[base+'run/STOP.json']==dict(identity=identity,normal_stop=True)
        model=selected[base+'run/MODEL_READY.json'];assert model['identity']==identity and model['Model_READY']and model['actual_owner_count']==16
        late.append(dict(slot=slot,MODEL_READY_present_after_rootSTOP=True,not_a_live_public_service_qualification=True))
        for stage in range(2):
            for rank in range(8):
                closed=selected[base+'run/closed-%d-%d.json'%(stage,rank)];life=selected[base+'run/lifecycle-%d-%d.json'%(stage,rank)]
                assert closed['identity']==identity and closed['normal']and closed['post_close_Torch_liveness']
                assert life['completed']and life['normal']and life['destroy_attempts']==life['shutdown_attempts']==1
                assert life['destroy_RC']==life['shutdown_RC']==0 and not life['cleanup_errors']and life['primary_error']is None
    assert post['CPU_post_passed']and all(p['returncode']==0 for p in post['post'])
    obs=[v for p in post['post']for v in p['observations']]
    assert len(obs)==len({v['node']for v in obs})==16
    assert all(v['CPU_only']and not v['GPU_initialized']and not v['kfd_pids']and not v['matching_component_processes']and not v['expected_PIDs_still_exist']and v['ports_bindable']=={'43843':True}for v in obs)
    accounts={r.split('|')[0]:r.split('|')for r in post['original_GPU_srun_and_keeper_accounting']['stdout'].splitlines()if'|'in r}
    assert all(accounts[s][2:4]==['COMPLETED','0:0']for s in GPU)
    assert all(accounts[s][2:4]==['CANCELLED','0:9']for s in CPU)
    def seconds(v):
        day,clock=(v.split('-',1)if'-'in v else('0',v));h,m,s=map(int,clock.split(':'));return int(day)*86400+h*3600+m*60+s
    residence=16*sum(seconds(accounts[s][4])for s in GPU)
    return dict(audit_passed=True,scope='All8 originalstartup infrastructure failure and exactcleanup, not Model or qualitypass',archive_SHA256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),verified_original_files=count,original_native_SHA256=NATIVE,normal_GPU_steps=GPU,normal_destroy_shutdown_owners=128,independent_post_nodes=16,KFD_PID_and_port_clean=True,original_keeper_steps=CPU,keepers_intentionally_CANCELLED_0_9=True,GPU_TERM_or_KILL_used=False,original_HTTP_completed=0,original_Native_request_completed=0,planned_questions=256,planned_HTTP=512,missing_or_unexecuted_HTTP=512,accuracy=None,new_speedup=None,Model_quality_qualified=False,Model_lifecycle_wholeplan_qualified=False,late_MODEL_READY=late,all16card_group_residence_s=residence,all_GPU_cardhours=residence/3600,GPU_busy_unknown=True,root_cause='UNKNOWN; ModelREADY was absent beforeStop and present afterStop, no Pythonstack obtained',next='Immutable boundedstartupdiagnostic on atmostone16card group, preservefull256/512goal; no automatic8group reload')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',required=True);a=p.parse_args();q=audit(a.archive)
    Path(a.output).write_text(json.dumps(q,indent=2)+'\n',encoding='utf-8');print(json.dumps(q,indent=2))
