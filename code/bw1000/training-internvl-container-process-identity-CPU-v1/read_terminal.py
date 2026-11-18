"""Pre-registered original metadata-only diagnostic consumer. No process relaunch."""
import hashlib,json,math,os,signal,time
from pathlib import Path
RUN=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-internvl-container-process-identity-CPU-v1')
def consume(context,read_epoch):
    assert math.isfinite(read_epoch) and 0<=time.time()-read_epoch<20
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,read_epoch+25-time.time())
    pins={};total=0
    def load(name,optional=False):
        nonlocal total
        p=RUN/name
        if optional and not p.exists():return None
        a=p.lstat();assert p.is_file() and not p.is_symlink() and a.st_size<=262144
        raw=p.read_bytes();z=p.lstat();assert all(getattr(a,k)==getattr(z,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))
        total+=len(raw);assert total<=1048576;pins[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest());return json.loads(raw)
    mf=load('source/SOURCE_FILES.json');claim=load('source/CLAIM.json')
    assert pins['source/SOURCE_FILES.json']['SHA256']==context['source_manifest_SHA256'] and pins['source/CLAIM.json']==mf['files']['CLAIM.json']
    assert claim['root_GO'] is claim['CPU_GO'] is claim['READ_GO'] is True and claim['GPU_GO'] is claim['Model_GO'] is False
    install=load('install-intent.json');out=load('outcome.json');state=load('control-state.json')
    for r in (install,out,state):
        for k in ('publication','request_id','epoch','deadline','source_manifest_SHA256'):assert r[k]==context[k]
    assert context['deadline']==context['epoch']+120
    node=load('node-result.json',True);observation=load('identity-observations.json',True);anchor=load('held-child.json',True);ids=load('owned-identities.json',True)
    handshake=load('handshake.json',True);done=load('handshake-done.json',True)
    row=out['accounting'];terminal=bool(row and row[1]==RUN.name and row[2].split()[0] in ('COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'))
    verified=False
    if out['diagnostic_receipt_verified']:
        assert terminal and row[0].startswith('1087656.') and row[2:4]==['COMPLETED','0:0'] and row[5]=='a14r3n04' and set(row[6].split(','))=={'cpu=2','mem=2G','node=1'}
        assert node==out['node'] and node['step']==row[0] and node['epoch']==context['epoch'] and node['deadline']==context['deadline']
        assert node['diagnostic_complete'] and node['held_cleanup']['reaped'] and not node['remaining_known_owned'] and not node['unknown_owned_liveness']
        assert out['held_srun']['reaped'] and not out['held_srun']['termination_needed'] and out['failure'] is None
        assert observation and observation['records'] and node['observed_records']==len(observation['records']) and observation['no_RSS'] and observation['no_global_proc_scan']
        assert anchor['PID']==anchor['PGID'] and anchor['start_new_session'] and anchor['supervisor']==node['owner']==observation['owner']
        assert all(node[k]==0 for k in ('worker_calls','Torch_calls','model_calls','weight_body_reads','RSS_reads'))
        if node['frontend_ready']:
            assert node['diagnostic_rejection'] is None and not node['held_cleanup']['termination_needed'] and done and done['Torch_imported'] is False
            assert all(r['accepted'] and r['error'] is None and r['UID_values']==[r['expected_UID']]*4 for r in observation['records'])
        verified=True
    result=dict(independent_diagnostic_verified=verified,original_terminal=terminal,actual_accounting=row,
        frontend_ready=bool(verified and node['frontend_ready']),diagnostic_rejection=node.get('diagnostic_rejection') if node else None,
        identity_records=observation.get('records') if observation else None,metadata_pins=pins,metadata_read_bytes=total,
        original_failure_preserved=True,UID_waiver=False,RSS_reads=0,GPU_GO=False,Model_GO=False,training_GO=False,
        current_cross_host_liveness='NOT_QUERIED_UNKNOWN',identity_transition_cause='REQUIRES_ACTUAL_RECORDED_FIELDS_NOT_INFERRED',replay_allowed=False)
    assert len(json.dumps(result).encode())<=262144;signal.setitimer(signal.ITIMER_REAL,0);return result
if __name__=='__main__':print(json.dumps(consume(CONTEXT,READ_EPOCH),sort_keys=True))
