"""One read-only original InternVL terminal/provenance/fresh-post observation.

No training, cleanup, torch import, artifact load/download or full artifact hash.
"""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

G = Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
S = G/'code/bw1000/training-internvl-A100-first-updates-v1'
REQUEST = '031e04f24efa429595364e6f0da1e5b6'
H = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
transport = json.loads((S/'transport-outcome.json').read_bytes())
assert transport['request_id'] == REQUEST and transport['transport_rc'] == 0
assert transport['remote_source_installed_acknowledged']
bound = dict(request=REQUEST, claim=transport['claim'],
    original_pins={n:row['SHA256'] for n,row in transport['value'].items() if isinstance(row,dict) and 'SHA256' in row},
    source_files=json.loads((S/'SOURCE_FILES.json').read_bytes()),
    binding=json.loads((S/'GPU_BINDING.json').read_bytes()),
    framework=json.loads((S/'FRAMEWORK_FILES.json').read_bytes()),
    installed_native=json.loads((S/'INSTALLED_SOURCE_FILES.json').read_bytes()))
REMOTE = r'''
from pathlib import Path
import datetime, hashlib, importlib.metadata as md, json, math, os, socket, subprocess, time
B=json.loads(BOUND_JSON)
R=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1')
epoch=time.monotonic(); end=epoch+75
out=dict(request_id=B['request'],claim=B['claim'],read_only=True,GPU_launched=False,
         capture_bytes_exported=0,artifact_hashes_recomputed=False,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks={},originals={})
def deadline():
    if time.monotonic()>=end:raise TimeoutError('original reader deadline')
def sha(p):
    deadline();return hashlib.sha256(p.read_bytes()).hexdigest()
def read(rel):
    deadline();p=R/rel;assert p.is_file() and not p.is_symlink() and p.stat().st_size<=2*1024**2
    raw=p.read_bytes();out['originals'][rel]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
    if rel in B['original_pins']:assert out['originals'][rel]['SHA256']==B['original_pins'][rel]
    return json.loads(raw)
def finite(v):
    if isinstance(v,float):return math.isfinite(v)
    if isinstance(v,list):return all(finite(x) for x in v)
    if isinstance(v,dict):return all(finite(x) for x in v.values())
    return True
def numeric(v):
    if isinstance(v,(int,float,bool)) or v is None:return v
    if isinstance(v,list):return [numeric(x) for x in v]
    if isinstance(v,dict):return {k:numeric(x) for k,x in v.items() if not any(t in k.lower() for t in ('command','environment','hostname','path','argv'))}
    return v if isinstance(v,str) and (v in ('T03.sft','cuda','bf16','bfloat16','COMPLETED') or len(v)<25 and not any(c in v for c in ('/','\\',':','@'))) else '[string omitted]'
try:
    docs={n:read(n) for n in ('install-intent.json','installed.json','supervisor-intent.json','gpu-intent.json','outcome.json','outer-result.json')}
    assert all(v['request_id']==B['request'] and v['claim']==B['claim'] for v in docs.values())
    outer,term,gpu,installed=docs['outer-result.json'],docs['outcome.json'],docs['gpu-intent.json'],docs['installed.json']
    assert outer['supervisor_rc']==0 and term['native_returncode']==0 and term['keeper_passed'] and not term['timed_out'] and term['controller_error'] is None
    assert term['cleanup']==dict(owned_processes_remaining=0,termination_needed=False)
    assert term['native25_rank_records']==term['capture_rank_records']==8
    assert gpu['framework_files']==349 and gpu['framework_commit']=='b715b6b52f28b24738b41388ca3d68a0e15e301f' and gpu['original25_preserved'] and gpu['GPU_cards']==8
    assert gpu['original_input_files_verified']==67
    for n,pin in installed['files'].items():assert sha(R/'src'/n)==pin
    fm=read('src/FRAMEWORK_FILES.json');assert fm==B['framework'] and len(fm['files'])==349
    for n,pin in fm['files'].items():
        p=R/'framework'/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
    for n,pin in B['installed_native'].items():
        dist={'transformers':'transformers','swift':'ms-swift','peft':'peft'}[n.split('/')[0]]
        p=Path(md.distribution(dist).locate_file(n));assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
    out['checks']['provenance_framework349_installed_providers']=True
    out['terminal']={k:term[k] for k in ('native_returncode','timed_out','controller_error','wall_seconds','keeper_passed','cleanup','native25_rank_records','capture_rank_records')}
    expected={'initial.pt'}|{'forward-%02d-%s.pt'%(i,kind) for i in range(1,9) for kind in ('input','output')}|{'update-%02d-%s.pt'%(i,kind) for i in (1,2) for kind in ('preclip','postclip','before-optimizer','after-optimizer')}
    ranks=[]
    for rank in range(8):
        native=read('train/nvidia-baseline/rank-%02d.json'%rank)
        capture=read('capture/rank-%02d/receipt.json'%rank)
        completed=read('capture/rank-%02d/native25-complete.json'%rank)
        assert completed==dict(rank=rank,global_step=25)
        assert capture['state']=='TWO_NATIVE_OPTIMIZER_UPDATES_CAPTURED' and capture['rank']==rank and capture['world_size']==8
        assert capture['completed_global_step']==2 and capture['micro_forwards']==8 and capture['optimizer_updates']==2 and capture['clip_calls']==2
        files=capture['files'];assert len(files)==25 and {f['name'] for f in files}==expected
        for f in files:
            p=R/('capture/rank-%02d'%rank)/f['name'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==f['bytes'] and 0<f['bytes']<=2*1024**3-1
            assert len(f['SHA256'])==64 and all(c in '0123456789abcdef' for c in f['SHA256'])
        total=sum(f['bytes'] for f in files);assert total<=8*1024**3 and finite(native)
        ranks.append(dict(rank=rank,native_metadata=numeric(native),native_metadata_finite=True,completed_global_step=25,
            first_capture_global_step=2,optimizer_updates=2,micro_forwards=8,capture_files=25,capture_bytes=total,
            largest_file_bytes=max(f['bytes'] for f in files),file_stats_match_receipt=True,
            artifact_SHA_pins_present=True,artifact_full_SHA_rechecked=False))
    out['ranks']=ranks;out['checks']['eight_rank_native25_and_first2_receipts_and_file_stats']=True
    # Only the exact previously recorded owner identities, never a /proc scan.
    identities={int(k):v for k,v in term['owners'].items()}
    for doc in (docs['install-intent.json'].get('owner',{}),docs['supervisor-intent.json']):
        if 'PID' in doc:identities[int(doc['PID'])]=doc
    states=[]
    for pid,v in identities.items():
        try:
            s=(Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()
            match=int(s[19])==v['startticks'];state=s[0]
            states.append(dict(PID=pid,same_startticks=match,state=state,live_original=match and state!='Z'))
        except FileNotFoundError:states.append(dict(PID=pid,exists=False,live_original=False))
    out['known_owners']=dict(count=len(states),live_original=[x for x in states if x['live_original']],absent=sum(x.get('exists') is False for x in states),reused_PID_count=sum(x.get('same_startticks') is False for x in states),zombie_original_count=sum(bool(x.get('same_startticks')) and x.get('state')=='Z' for x in states))
    def smi(args):
        deadline();return subprocess.check_output(['nvidia-smi',*args],text=True,timeout=min(12,end-time.monotonic())).strip().splitlines()
    cards=[[v.strip() for v in row.split(',')] for row in smi(['--query-gpu=index,uuid,pci.bus_id,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'])]
    apps=smi(['--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'])
    match=len(cards)==8 and {r[0]:r[1] for r in cards}==B['binding']['UUID_by_index'] and {r[0]:r[2] for r in cards}==B['binding']['PCI_by_index']
    bindable=False
    try:
        with socket.socket() as s:s.bind(('127.0.0.1',29901));bindable=True
    except OSError:pass
    out['freshpost']=dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),UUID_PCI_match=match,cards=cards,compute_apps=apps,port29901_bindable=bindable,
        all8_idle=match and not apps and all(r[3]=='NVIDIA A100-SXM4-40GB' and int(r[4])<=16 and int(r[5])==0 for r in cards))
    out['minimum_terminal_freshpost_pass']=bool(out['freshpost']['all8_idle'] and bindable and not out['known_owners']['live_original'])
    out['state']='ORIGINAL_TERMINAL_AND_FRESHPOST_PASS' if out['minimum_terminal_freshpost_pass'] else 'FRESHPOST_NOT_CLEAR'
    out['finite_boundary']='Captured tensor cpu() checks all floating tensors finite at original save; current review verifies receipt/source only, no deserialization; native JSON numbers finite.'
except Exception as error:
    out.update(state='INDEPENDENT_READER_FAILED',error_type=type(error).__name__,minimum_terminal_freshpost_pass=False)
    import traceback
    out['reader_failure_line']=traceback.extract_tb(error.__traceback__)[-1].lineno
out['elapsed_seconds']=time.monotonic()-epoch
print(json.dumps(out,ensure_ascii=False,allow_nan=False))
'''
if __name__ == '__main__':
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')
    D=G/'local-only'/('internvl-A100-original-terminal-'+stamp+'-astra105');D.mkdir(exist_ok=False)
    payload=('BOUND_JSON='+repr(json.dumps(bound))+'\n').encode()+REMOTE.encode()
    def save(n,v):
        with (D/n).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
    save('intent.json',dict(request_id=REQUEST,reader_SHA256=H(Path(__file__)),payload_SHA256=hashlib.sha256(payload).hexdigest(),original_transport_SHA256=H(S/'transport-outcome.json'),one_read_only_connection=True,artifact_bytes_exported=0))
    config=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config')
    argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','StrictHostKeyChecking=yes','-F',str(config),'a100-27','/home/whj/nvidia-baseline/env-vlm/bin/python -B -u -']
    try:
        p=subprocess.run(argv,input=payload,capture_output=True,timeout=85)
        rc,stdout,stderr=p.returncode,p.stdout,p.stderr
    except subprocess.TimeoutExpired as e:rc,stdout,stderr=None,e.stdout or b'',e.stderr or b''
    value=json.loads(stdout) if rc==0 else None
    receipt=dict(transport_rc=rc,stdout_SHA256=hashlib.sha256(stdout).hexdigest(),stderr_SHA256=hashlib.sha256(stderr).hexdigest(),stderr_bytes=len(stderr),value=value)
    save('receipt.json',receipt)
    print(json.dumps(dict(directory=str(D),receipt_SHA256=H(D/'receipt.json'),value=value),ensure_ascii=False,indent=2))
