"""Run the pinned Swift sampler projection in one zero-GRES production-runtime step."""
from base64 import b64decode
from hashlib import sha256
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import time

ROOT=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
RESULTS=ROOT/'results'
LABEL='training-gemma-r45-swift-sampler-cpu-projection-v1'
PARENT='1103606'
NODE='f13r1n12'
PLAN=RESULTS/'training-gemma-usgs72-fresh-initial-domain25-v1/continuous-plan.json'
INDEX=RESULTS/'training-gemma-usgs-r39-two-updates-v1/6d14e9df9832432b97f1f46f032b49f3'
TRACE=RESULTS/'training-gemma-usgs-r44-live-sampler-diagnostic-v1'
COLS='JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End'
PIN='070caf5a6ae0a0158d7d2fb127f632b2f30d4e3a929ac5004c783b92d3f0baa9'


def require(ok,why):
    if not ok:
        raise ValueError(why)


def query(*argv):
    data=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,check=True,timeout=8)
    require(not data.stderr and len(data.stdout)<=1048576,'G45_HOST_BOUNDED_AUTHORITY')
    return data.stdout.decode().splitlines()


def fields(line):
    return dict(q.split('=',1) for q in line.split() if '=' in q)


def hosts(value):
    if re.fullmatch(r'f[0-9]{2}r[0-9]+n[0-9]{2}',value):
        return [value]
    return query('scontrol','show','hostnames',value)


def accounting(selector):
    rows=[line.split('|') for line in query('sacct','-nP','-j',selector,'-o',COLS)]
    require(all(len(x)==9 for x in rows),'G45_HOST_NINE_ACCOUNTING_FIELDS')
    return rows


def put(path,raw):
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def pin(path,cap,expected):
    m=path.lstat()
    require(path.resolve(strict=True)==path and stat.S_ISREG(m.st_mode) and 0<m.st_size<=cap,'G45_EXACT_PRIOR_FILE')
    with path.open('rb') as f:
        raw=f.read(cap+1)
    require(len(raw)==m.st_size and sha256(raw).hexdigest()==expected,'G45_PRIOR_FILE_HASH')
    return raw


def main(a):
    request=a['request']
    require(re.fullmatch(r'[0-9a-f]{32}',request) and re.fullmatch(r'[0-9a-f]{40}',a['publication'])
            and os.getuid()==154459,'G45_HOST_BOUND_IDENTITY')
    source=b64decode(a['source'],validate=True)
    require(len(source)==10606 and sha256(source).hexdigest()==PIN,'G45_EXACT_NEW_SOURCE')
    root=RESULTS/LABEL
    with (RESULTS/'.training80-dcu-standby.lock').open('rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not os.path.lexists(root) and RESULTS.resolve(strict=True)==RESULTS,'G45_ORIGINAL_NEW_NAMESPACE')
        old_out=json.loads(pin(TRACE/'outcome.json',65536,'00d435e47c82bfae723e9a67fa61b00230ca0baa8a332e5d3e1c50248b8142a5'))
        require(old_out['release_permitted'] is True and old_out['diagnostic_completed'] is False,'G45_PRIOR_OWNER_BOUNDARY')
        expected={'1103606.3406':('COMPLETED','0:0'),'1103606.3465':('FAILED','1:0'),'1103606.3466':('COMPLETED','0:0')}
        original=[r for r in accounting(','.join(expected)) if r[0] in expected]
        require(len(original)==3 and {x[0] for x in original}==set(expected)
                and all(tuple(x[2:4])==expected[x[0]] and x[5]==NODE for x in original),'G45_ACTUAL_ORIGINALS')
        [j]=list(map(fields,query('scontrol','-o','show','job',PARENT)))
        require(j['JobId']==PARENT and j['JobState']=='RUNNING' and j['UserId'].endswith('(154459)')
                and NODE in hosts(j['NodeList']),'G45_CURRENT_PARENT')
        for line in query('squeue','--steps','-h','-j',PARENT,'-o','%i|%N'):
            h,n=line.split('|')
            if re.fullmatch(PARENT+r'\.[0-9]+',h):
                require(NODE not in hosts(n),'G45_CURRENT_QUEUE_HAS_NUMBERED')
        for d in map(fields,query('scontrol','-o','show','step',PARENT)):
            if re.fullmatch(PARENT+r'\.[0-9]+',d.get('StepId','')) and d['State'] not in (
                    'COMPLETED','FAILED','TIMEOUT','CANCELLED','NODE_FAIL'):
                require(NODE not in hosts(d['NodeList']),'G45_CURRENT_CONTROLLER_HAS_NUMBERED')
        require(not any(x[1]==LABEL for x in accounting(PARENT)),'G45_ORIGINAL_LABEL_ALREADY_PRESENT')
        root.mkdir(mode=0o700)
        (root/'tmp').mkdir(mode=0o700)
        put(root/'probe.py',source)
        epoch=time.time()
        identity=dict(request=request,publication=a['publication'],epoch=epoch,deadline=epoch+120,GPU_requested=0)
        runner=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--no-home','--pwd','/']
        for host,inside,mode in ((root,'/g45','rw'),(INDEX,'/gemma-r45-index','ro'),(PLAN,'/gemma-r45-plan.json','ro'),
                (RESULTS/'training-vlm-swift-offline-runtime-v1','/swift-runtime','ro'),
                (RESULTS/'training-minicpm-hygon-processor-r2/cpu/runtime-site','/minicpm-transformers','ro')):
            runner+=['--bind',f'{host}:{inside}:{mode}']
        shell=('set -euo pipefail; source /opt/dtk/env.sh; '
               'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/minicpm-transformers:/swift-runtime/site '
               'HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 TMPDIR=/g45/tmp TMP=/g45/tmp TEMP=/g45/tmp; '
               'exec '+shlex.join(['python3','-B','/g45/probe.py','--epoch',str(epoch),'--request',request]))
        run=['srun','--jobid='+PARENT,'--exact','--immediate=5','--nodes=1','--ntasks=1','--cpus-per-task=4',
             '--mem=8192M','--nodelist='+NODE,'--gres=none','--time=00:02:00','--job-name='+LABEL]+runner
        run+=[str(ROOT/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',shell]
        put(root/'intent.json',(json.dumps(identity|dict(source_SHA256=PIN),sort_keys=True)+'\n').encode())
        with (root/'stdout').open('xb') as out,(root/'stderr').open('xb') as err:
            child=subprocess.Popen(run,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
        put(root/'client.json',(json.dumps(identity|dict(client_PID=child.pid),sort_keys=True)+'\n').encode())
    try:
        rc=child.wait(timeout=36)
    except subprocess.TimeoutExpired:
        return identity|dict(state='ORIGINAL_OUTCOME_UNKNOWN',replay=False)
    row=[]
    for attempt in range(3):
        row=[r for r in accounting(PARENT) if r[1]==LABEL]
        if row and row[0][2] not in ('RUNNING','PENDING','COMPLETING'):
            break
        time.sleep(.25)
    with (root/'stdout').open('rb') as f:raw=f.read(16385)
    with (root/'stderr').open('rb') as f:err=f.read(16385)
    require(max(len(raw),len(err))<=16384,'G45_HOST_OUTPUT_BOUND')
    try:
        value=json.loads(raw) if raw else None
    except ValueError:
        value=None
    good=bool(rc==0 and value and len(row)==1 and row[0][0]==value['step'] and row[0][2:4]==['COMPLETED','0:0'] and row[0][5]==NODE
              and set(row[0][6].split(',')) in ({'cpu=4','mem=8G','node=1'},{'cpu=4','mem=8192M','node=1'})
              and value['epoch']==epoch and value['request']==request and value['completed'] is True and value['compared_slots']==64
              and value['actual_GPU']==value['optimizer_updates']==0 and value['native_live_sampler_iterated'] is False)
    return identity|dict(state='ACTUAL_CPU_C0_EXACT_PROJECTION' if good else 'ORIGINAL_CHECK_UNQUALIFIED',
                         original_accounting=row,stdout_SHA256=sha256(raw).hexdigest(),stderr_SHA256=sha256(err).hexdigest(),
                         rc=rc,projection=value if good else None,replay=False,training_ready_added=0)


if __name__=='__main__':
    a=json.load(sys.stdin)
    try:
        observed=main(a)
    except (OSError,ValueError,TypeError,KeyError,subprocess.SubprocessError) as error:
        code=str(error)
        observed=dict(request=a['request'],state='ORIGINAL_CHECK_UNQUALIFIED',error_class=type(error).__name__,
                      code=code if re.fullmatch(r'G45_[A-Z_0-9]+',code) else None,training_ready_added=0)
    print(json.dumps(observed,sort_keys=True))
