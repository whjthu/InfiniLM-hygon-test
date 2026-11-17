"""Shared original-epoch and owned-process primitives; no SDK imports."""
import hashlib,json,math,os,re,signal,socket,stat,subprocess,time
from pathlib import Path
from slurm_accounting import wait_terminal
R=Path('/refactor') if str(Path(__file__).resolve()).startswith('/refactor/') else Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LABEL='GLM474-history-value-only-device-validation-GPU-v1'
W=R/'results'/LABEL;D=W/'src';PARENT='1103606';GPU_NODE='f13r1n05';KEEPER_NODE='f13r1n07'
H=lambda b:hashlib.sha256(b).hexdigest()
def need(ok,code):
    if not ok:raise ValueError(code)
def load(p):return json.loads(Path(p).read_bytes())
def save(p,value,exclusive=False):
    p=Path(p);tmp=p.with_name(p.name+'.tmp-'+str(os.getpid()))
    if exclusive:need(not p.exists(),'ORIGINAL_FILE_EXISTS')
    with tmp.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p);fd=os.open(str(p.parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def owner(pid):
    row=Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()
    return dict(PID=pid,PGID=int(row[2]),startticks=int(row[19]),host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),UID=os.getuid())
class Budget:
    def __init__(self,epoch):self.epoch=epoch;self.anchor=time.monotonic();self.available=epoch+660-time.time();need(0<self.available<=660,'ORIGINAL_EPOCH')
    def left(self,cap=660,reserve=0):
        value=min(self.epoch+cap-time.time(),self.available-(time.monotonic()-self.anchor))-reserve;need(math.isfinite(value) and value>0,'ORIGINAL_EPOCH_EXHAUSTED');return value
def cleanup(proc,record):
    # Only called with the Popen held here, created with start_new_session=True.
    # Until wait/poll reaps this direct child its PID cannot be reused. This
    # ownership anchor survives a /proc read or owner-receipt write failure.
    result=dict(anchor='held_Popen_new_session',owner_record_verified=None)
    if proc is None:return result
    if proc.poll() is None:
        if record:
            try:
                now=owner(proc.pid);need(now['PGID']==proc.pid and all(now[k]==record[k] for k in ('PID','startticks','host','boot_id','UID')),'OWN_CHILD_IDENTITY')
                result['owner_record_verified']=True
            except BaseException as error:
                result.update(owner_record_verified=False,owner_verification_error=type(error).__name__)
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait(timeout=3)
    else:proc.wait(timeout=0)
    return result
def run_owned(argv,log,budget,cap,record_path,preserve_env=False):
    proc=None;record={};rc=None;forced=False;error=None;cleanup_record=None
    with Path(log).open('xb') as f:
        try:
            budget.left(cap,5);proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ) if preserve_env else clean_env())
            record=owner(proc.pid);save(record_path,record,True);rc=proc.wait(timeout=budget.left(cap,5))
        except BaseException as e:forced=True;error=type(e).__name__
        finally:
            if proc is not None:
                if proc.poll() is None:forced=True
                cleanup_record=cleanup(proc,record)
    return dict(rc=rc,forced=forced,error_type=error,owned_reaped=proc is not None and proc.returncode is not None,owner=record,cleanup=cleanup_record)
def clean_env():
    return {k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','PMI_','PMIX_','CUDA_','HIP_','ROCR_','GPU_')) or k in ('SLURM_CONF','SLURM_CONF_SERVER')}
def query(budget,*argv):return subprocess.check_output(argv,text=True,timeout=min(10,budget.left())).splitlines()
def account(budget,label):
    rows=[line.split('|') for line in query(budget,'sacct','-n','-P','-j',PARENT,'-o','JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES') if '|'+label+'|' in line]
    need(len(rows)<=1 and all(len(r)==7 and re.fullmatch(PARENT+r'\.[0-9]+',r[0]) for r in rows),'EXACT_ACCOUNTING');return rows[0] if rows else None
def step(budget,phase,node,cpu,mem,gpus,cap,entry):
    nodes=node.split(',');label=LABEL+'-'+phase;need(account(budget,label) is None,'STEP_NO_REPLAY');phase_limit=60 if phase in ('pre','post') else cap;seconds=60*(min(int(budget.left(cap,15)),phase_limit)//60);need(seconds>=60,'SLURM_MINUTE_FLOOR')
    argv=['srun','--jobid='+PARENT,'--job-name='+label,'--exact','--immediate=5','--nodes='+str(len(nodes)),'--ntasks='+str(len(nodes)),'--ntasks-per-node=1','--distribution=block:block','--nodelist='+node,'--cpus-per-task='+str(cpu),'--mem='+str(mem)+'M','--gres='+('dcu:'+str(gpus) if gpus else 'none'),'--time='+str(seconds//60),*entry]
    receipt=dict(phase=phase,label=label,exec_request_attempted=False,epoch=budget.epoch,state='NOT_SUBMITTED',argv=argv);path=W/(phase+'-receipt.json');save(path,receipt,True)
    receipt.update(exec_request_attempted=True,state='SUBMISSION_POSSIBLE');save(path,receipt)
    receipt.update(run_owned(argv,W/(phase+'.log'),budget,cap,W/(phase+'-srun-owner.json')))
    terminal=wait_terminal(lambda:account(budget,label),time.monotonic()+min(24,budget.left(660,5)),interval=1,max_observations=16);row=terminal['row'];receipt.update(accounting=terminal,row=row,terminal=terminal['terminal_visible'],state='TERMINAL' if terminal['terminal_visible'] else 'UNKNOWN')
    tres=dict(x.split('=',1) for x in row[6].split(',') if '=' in x) if row else {}
    actual_nodes=set(query(budget,'scontrol','show','hostnames',row[5])) if row else set()
    receipt['passed']=bool(row and row[2:4]==['COMPLETED','0:0'] and actual_nodes==set(nodes) and tres.get('cpu')==str(cpu*len(nodes)) and tres.get('gres/dcu','0')==str(gpus*len(nodes)) and receipt['rc']==0 and not receipt['forced']);save(path,receipt);return receipt
def file_pin(path,pin,cap=4*1024**2):
    p=Path(path);s=p.stat();need(stat.S_ISREG(s.st_mode) and 0<=s.st_size<=cap and (pin['bytes'] is None or s.st_size==pin['bytes']),'PIN_FILE_BOUND')
    with p.open('rb') as f:a=os.fstat(f.fileno());raw=f.read(cap+1);z=os.fstat(f.fileno())
    identity=lambda t:(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns,t.st_ctime_ns)
    need(identity(s)==identity(a)==identity(z)==identity(p.stat()) and H(raw)==pin['SHA256'],'STABLE_FILE_PIN');return raw
def qualify_cpu(host=True):
    e=load(D/'CPU_EVIDENCE.json');need(e['qualified'] is True and all(e[k] is not None for k in ('CPU_publication','CPU_request_id','CPU_step','independent_static_evidence','source_manifest_SHA256','source_claim_SHA256')),'CPU_V3_EVIDENCE_PENDING')
    def read_pin(pin):
        path=pin['path'];need(path.startswith('/refactor/results/'),'CPU_EVIDENCE_PATH');p=str(R)+path[len('/refactor'):] if host else path;return file_pin(p,pin)
    out=json.loads(read_pin(e['outcome']));build=json.loads(read_pin(e['build']));pin=e['independent_static_evidence'];raw=__import__('base64').b64decode(pin['raw_b64'],validate=True);need(len(raw)==pin['bytes'] and H(raw)==pin['SHA256'],'INDEPENDENT_CPU_STATIC_PIN');static=json.loads(raw)
    need(out['accounting']['row'][0]==e['CPU_step'] and out['accounting']['row'][2:4]==['COMPLETED','0:0'] and out['rc']==0,'CPU_TERMINAL')
    need(build['step']==e['CPU_step'] and build['passed'] is True and build['actual']==dict(freshM=1,HOST_compile=1,GPU_compile=0,objects=1,links=1) and static['passed'] is True and static['CPU_static_qualified'] is True and static['actual_accounting'][0]==e['CPU_step'],'CPU_STATIC_QUALIFICATION')
    need(out['epoch']==e['CPU_epoch'] and static['actual_accounting']==out['accounting']['row'] and static['files']['build-result.json']['SHA256']==e['build']['SHA256'] and static['files']['src/SOURCE_FILES.json']['SHA256']==e['source_manifest_SHA256'] and static['files']['src/CLAIM.json']['SHA256']==e['source_claim_SHA256'] and static['files']['src/component.cpp']['SHA256']==e['corrected_CPP_SHA256'],'CPU_PROVENANCE_CROSS_BINDING')
    base='/refactor/results/GLM474-history-value-only-component-CPU-v3/'
    for name in ('src/SOURCE_FILES.json','src/CLAIM.json','src/component.cpp'):
        pin=dict(static['files'][name],path=base+name);read_pin(pin)
    install_path=(str(R)+base[len('/refactor'):]) if host else base
    install_file=Path(install_path)/'install-intent.json';need(install_file.stat().st_size<=16384,'CPU_INSTALL_BOUND');install=json.loads(install_file.read_bytes());need(all(install[k]==e[v] for k,v in [('publication','CPU_publication'),('request_id','CPU_request_id'),('epoch','CPU_epoch'),('source_manifest_SHA256','source_manifest_SHA256'),('claim_SHA256','source_claim_SHA256')]),'CPU_ORIGINAL_INSTALL_IDENTITY')
    need(build['library']['SHA256']==e['DSO']['SHA256'] and build['library']['bytes']==e['DSO']['bytes'],'CPU_DSO_RECORD');read_pin(e['DSO']);read_pin(e['parent_B']);return e,build
