"""Dispatch one two-node collective with owned steps and post-device checks."""
import base64,hashlib,json,os,re,signal,socket,subprocess,sys,time
from pathlib import Path
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
PARENT='1103606'
NODES=['f13r1n05', 'f13r1n07']
LABEL='GLM474-TP16-native-collective-v1'
W=R/'results'/LABEL
epoch=time.time()
anchor=time.monotonic()
columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList']
def left():
    return min(epoch+300-time.time(),300-(time.monotonic()-anchor))
def query(args):
    return subprocess.check_output(args,text=True,timeout=min(12,left())).strip()
def rows(label):
    raw=query(['sacct','-n','-P','-j',PARENT,'--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'])
    return [r.split('|') for r in raw.splitlines() if '|'+label+'|' in r]
def save(name,value):
    with (W/name).open('x') as f:
        json.dump(value,f,indent=2)
        f.flush()
        os.fsync(f.fileno())
assert not W.exists() and all(not rows(LABEL+'-'+phase) for phase in ['pre','gpu','post'])
parent=query(['squeue','-h','-j',PARENT,'-o','%T|%D|%N']).split('|')
assert parent[:2]==['RUNNING','40']
assert set(NODES)<=set(query(['scontrol','show','hostnames',parent[2]]).splitlines())
for line in query(['squeue','--steps','-h','-j',PARENT,'-o','%i|%N']).splitlines():
    handle,nodes=line.split('|',1)
    if handle not in (PARENT+'.batch',PARENT+'.extern') and nodes not in ('','(null)'):
        assert not set(NODES).intersection(query(['scontrol','show','hostnames',nodes]).splitlines())
W.mkdir(mode=0o700)
stat=Path('/proc/self/stat').read_text().split(') ',1)[1].split()
save('intent.json',dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=epoch+300,work_deadline=epoch+180,
    nodes=NODES,owner=dict(host=socket.gethostname(),PID=os.getpid(),startticks=int(stat[19])),
    source_files={n:x['SHA256'] for n,x in FILES.items()}))
for folder in ['src','pre','post','ranks']:
    (W/folder).mkdir()
for name,pin in FILES.items():
    assert Path(name).name==name
    raw=base64.b64decode(pin['b64'],validate=True)
    assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
    (W/'src'/name).write_bytes(raw)
sys.path.insert(0,str(W/'src'))
from slurm_accounting import wait_terminal
def run(phase,timeout):
    gpu=phase=='gpu'
    label=LABEL+'-'+phase
    cmd=['srun','--jobid='+PARENT,'--job-name='+label,'--exact','--nodes=2','--ntasks='+('16' if gpu else '2'),'--ntasks-per-node='+('8' if gpu else '1'),
         '--distribution=block:block','--nodelist='+','.join(NODES),'--immediate=5','--kill-on-bad-exit=1',
         '--cpus-per-task='+('4' if gpu else '1'),'--mem='+('16384M' if gpu else '1024M'),
         '--gres='+('dcu:8' if gpu else 'none'),'--time='+('3' if gpu else '1'),
         '/usr/local/python3.12/bin/python3','-B',str(W/'src/host_entry.py'),str(W),'gpu-gate' if gpu else phase]
    if gpu:
        cmd=cmd[:cmd.index('/usr/local/python3.12/bin/python3')]
        scope='/refactor/results/'+LABEL
        shell='set -euo pipefail; source "$1/src/GPU_ENV.sh"; exec /usr/bin/python3 -B -u "$1/src/worker.py"'
        cmd+=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--rocm','--no-home','--pwd','/',
              '--bind',str(R)+':/refactor:ro','--bind',str(W)+':'+scope+':rw',
              '--bind','/opt/hyhal:/opt/hyhal:ro',str(R/'containers/dtk26-sandbox-r1'),
              '/bin/bash','--noprofile','--norc','-c',shell,'component',scope]
    timed_out=False
    with (W/(phase+'.log')).open('xb') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            st=Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split()
            save(phase+'-owner.json',dict(PID=child.pid,PGID=int(st[2]),startticks=int(st[19])))
            child.wait(timeout=min(timeout,left()-35))
        except subprocess.TimeoutExpired:
            timed_out=True
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait(timeout=5)
    def one_row():
        rs=rows(label)
        assert len(rs)<=1
        return rs[0] if rs else None
    terminal=wait_terminal(one_row,time.monotonic()+min(15,left()-5),interval=.5,max_observations=30)
    raw=(W/(phase+'.log')).read_bytes()
    assert len(raw)<2**20
    row=terminal['row']
    receipt=dict(terminal_visible=terminal['terminal_visible'],passed=bool(row and row[2:4]==['COMPLETED','0:0'] and set(query(['scontrol','show','hostnames',row[5]]).splitlines())==set(NODES) and child.returncode==0 and not timed_out),
                 row=row,returncode=child.returncode,timed_out=timed_out,log_SHA256=hashlib.sha256(raw).hexdigest(),
                 log_bytes=len(raw))
    save(phase+'-receipt.json',receipt)
    return receipt
pre=run('pre',35)
gpu=post=None
if pre['passed']:
    gpu=run('gpu',max(1,epoch+190-time.time()))
    if gpu['terminal_visible']:
        post=run('post',35)
rank_files=[W/'ranks'/f'{i:02d}.json' for i in range(16)]
device=None
if all(f.is_file() for f in rank_files):
    reports=[json.loads(f.read_bytes()) for f in rank_files]
    assert [r['rank'] for r in reports]==list(range(16))
    assert len({r['uid_SHA256'] for r in reports})==1
    assert all(r['passed'] and r['step']==reports[0]['step'] and r['node']==NODES[r['rank']//8] for r in reports)
    for node in NODES:
        selected=[r for r in reports if r['node']==node]
        pre_record=json.loads((W/'pre'/f'{node}.json').read_bytes())
        assert sorted(r['pci'] for r in selected)==pre_record['pci']
        assert len({r['kfd_id'] for r in selected})==8
    device=dict(passed=True,step=reports[0]['step'],ranks=reports,
                rank_SHA256=[hashlib.sha256(f.read_bytes()).hexdigest() for f in rank_files])
    save('device.json',device)
passed=bool(pre['passed'] and gpu and gpu['passed'] and post and post['passed'] and device
            and device['passed'] and device['step']==gpu['row'][0])
out=dict(request_id=REQUEST,claim=CLAIM,passed=passed,accounting=[r['row'] for r in (pre,gpu,post) if r and r['row']],
         accounting_columns=columns,pre=pre,gpu=gpu,post=post,device=device,
         wall_seconds=time.monotonic()-anchor,GPU_cards=16 if gpu else 0)
save('outcome.json',out)
print(json.dumps(out),flush=True)
