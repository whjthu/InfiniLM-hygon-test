"""Offline exact-byte audit; a normal GPU STOP is not public-entry success."""
import argparse,hashlib,json,tarfile
from pathlib import Path,PurePosixPath
def audit(path):
    files={}
    with tarfile.open(path,'r:gz')as tf:
        ms=tf.getmembers();assert len({m.name for m in ms})==len(ms)
        manifest=json.load(tf.extractfile('MANIFEST.json'));assert len(manifest['files'])+1==len(ms)
        for m in ms:
            assert m.isfile()and not PurePosixPath(m.name).is_absolute()and'..'not in PurePosixPath(m.name).parts
            if m.name=='MANIFEST.json':continue
            raw=tf.extractfile(m).read();ref=manifest['files'][m.name]
            assert len(raw)==m.size==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['SHA256'];files[m.name]=raw
    read=lambda n:json.loads(files[n]);w=read('watch.json');identity=read('run/FRONTEND.json')['identity']
    assert w['completed']and not w['passed']and w['actual_GPU']=='910376.879'and w['CPU_keeper']=='910407.941'and w['actual_client']is None
    assert w['client_launch_blocked']and not w['startup_accepted']and w['normal_STOP_published']and not w['forced_cleanup']
    assert w['startup_STOP_drain']['normal_GPU_STOP_drain']and'all16ready120=True'in w['error']
    assert identity['actual_step']==w['actual_GPU']and identity['gpu_start_epoch']==w['original_GPU_start_unix']==1793747525.260628
    assert read('run/STOP.json')==dict(identity=identity,normal_stop=True)
    for stage in range(2):
        masks=[]
        for rank in range(8):
            owner=read('run/owner-%d-%d.json'%(stage,rank));life=read('run/lifecycle-%d-%d.json'%(stage,rank));closed=read('run/closed-%d-%d.json'%(stage,rank));ready=read('run/ready-%d-%d.json'%(stage,rank))
            assert owner['actual_step']==w['actual_GPU']and owner['Native_SHA256']==owner['library_sha256']=='9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'
            assert owner['completed']and owner['worker_shutdown_completed']and owner['normal_shutdown']and owner['native_loop_drained']and owner['post_close_Torch_liveness']
            assert owner['attempted_Native']==owner['completed_Native']==0 and owner['service_requests']==[]
            assert ready['identity']==identity and ready['loaded']and ready['postload_4GiB']and ready['service_agreement']
            mine=[r for r in ready['affinity']['actual_owner_report']['owners']if r['device']==rank];assert len(mine)==1 and mine[0]['changes']==1
            mask=list(map(int,mine[0]['verified_mask'].split(',')));assert len(mask)==4;masks.extend(mask)
            assert life['completed']and life['normal']and life['destroy_attempts']==life['shutdown_attempts']==1 and life['destroy_RC']==life['shutdown_RC']==0 and not life['cleanup_errors']and life['primary_error']is None
            assert closed['identity']==identity and closed['normal']and closed['post_close_Torch_liveness']
        assert len(masks)==len(set(masks))==32
    post=w['post_after_failure'];assert len(post)==2 and len({r['node']for r in post})==2
    assert all(r['CPU_only']and not r['GPU_initialized']and not r['kfd_pids']and not r['matching_component_processes']and not r['expected_PIDs_still_exist']and r['ports_bindable']=={'43843':True}for r in post)
    accounts={r[0]:r for r in read('accounting.json')['rows']};assert accounts['910376.879'][2:4]==['COMPLETED','0:0']and accounts['910407.941'][2:4]==['FAILED','1:0']and accounts['910376.880'][2:4]==['COMPLETED','0:0']
    snapshot=files['startup-snapshot.json'];assert hashlib.sha256(snapshot).hexdigest()=='64ff639e5a1430d637d491195563f67046ffbd31cd17819ffb0b8e7f4d724919'
    snap=json.loads(snapshot);assert len(snap['files'])==21
    for f in snap['files']:
        raw=f['rawtext'].encode();assert len(raw)==f['bytes']and hashlib.sha256(raw).hexdigest()==f['SHA256']
    stacks=[f for f in snap['files']if f['path'].endswith('.txt')];assert len(stacks)==1
    text=stacks[0]['rawtext'];assert len(text.encode())==28302 and hashlib.sha256(text.encode()).hexdigest()=='a2bd4198ac06efe6e919299b011cc7df6872a9190eb52f8cb7600fbe646c0fd1'
    assert 'protocol.py", line 71 in wait'in text and'native_engine_bridge_r268.py", line 128 in take'in text
    return dict(audit_passed=True,archive_SHA256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),verified_raw_files=len(files),actual_GPU='910376.879',CPU_keeper='910407.941',GPU_state='COMPLETED0',keeper_state='FAILED1',independent_post='910376.880 COMPLETED0',normal_destroy_shutdown_owners=16,KFD_PID_port_clean_nodes=2,GPU_forced_cleanup=False,planned_HTTP=64,completed_HTTP=0,unexecuted_HTTP=64,global_original_quality_target_HTTP=512,accuracy=None,Model_entry_passed=False,Model_quality_passed=False,real_Python_stack_bytes=28302,real_Python_dump_count=text.count('Timeout (0:00:20)!'),actual_wait_location='protocol.wait while Path.exists false; actual missing name/path/view UNKNOWN',Native_C_or_GPU_stack_captured=False,late_Model_READY_present='run/MODEL_READY.json'in files,late_ready_is_not_entry_qualification=True,next='Bounded original-node same-namespace CPU read diagnostic; no automatic reload',new_performance_round=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',required=True);a=p.parse_args();q=audit(a.archive)
    Path(a.output).write_text(json.dumps(q,indent=2)+'\n',encoding='utf-8');print(json.dumps(q))
