"""Dispatch one two-node collective with owned steps and post-device checks."""
import base64,hashlib,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
PARENT='1103606'
NODES=['f12r2n10','f12r2n11']
LABEL='added-o01-rccl-loader-alias-v1'
W=R/'results'/LABEL
epoch=time.time()
anchor=time.monotonic()
columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList']
def left():
    return min(epoch+260-time.time(),260-(time.monotonic()-anchor))
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
save('intent.json',dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=epoch+260,work_deadline=epoch+165,
    nodes=NODES,owner=dict(host=socket.gethostname(),PID=os.getpid(),startticks=int(stat[19])),
    source_files={n:x['SHA256'] for n,x in FILES.items()}))
for folder in ['src','pre','gpu-gate','post','ranks','traces']:
    (W/folder).mkdir()
for name,pin in FILES.items():
    assert Path(name).name==name
    raw=base64.b64decode(pin['b64'],validate=True)
    assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
    (W/'src'/name).write_bytes(raw)
alias_dir=W/'verbs-aliases'
alias_dir.mkdir(mode=0o700)
for name in ('libibverbs.so','libibverbs.so.1'):
    (alias_dir/name).symlink_to('/rdma-provider-stack/libibverbs.so.1')
sys.path.insert(0,str(W/'src'))
from slurm_accounting import wait_terminal
def run(phase,timeout):
    gpu=phase=='gpu'
    label=LABEL+'-'+phase
    cmd=['srun','--jobid='+PARENT,'--job-name='+label,'--exact','--nodes=2','--ntasks=2','--ntasks-per-node=1',
         '--distribution=block:block','--nodelist='+','.join(NODES),'--immediate=5','--kill-on-bad-exit=0',
         '--cpus-per-task='+('4' if gpu else '1'),'--mem='+('8192M' if gpu else '1024M'),
         '--gres='+('dcu:1' if gpu else 'none'),'--time='+('3' if gpu else '1'),
         '/usr/local/python3.12/bin/python3','-B',str(W/'src/host_entry.py'),str(W),'gpu-gate' if gpu else phase]
    if gpu:
        shell='set -euo pipefail; source /opt/dtk/env.sh; export LD_LIBRARY_PATH="$1/verbs-aliases:/rdma-provider-stack:$LD_LIBRARY_PATH" RDMAV_DRIVERS=/rdma-provider-stack/libshca NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_DEBUG=INFO OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1; exec python3 -B -u "$1/src/worker.py" "$1"'
        cmd+=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--rocm','--no-home',
              '--bind',str(W)+':'+str(W),'--bind',str(R/'build-deps/rdma-host-stack')+':/rdma-provider-stack:ro','--bind','/opt/hyhal:/opt/hyhal:ro',
              str(R/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',shell,'collective',str(W)]
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
    receipt=dict(passed=bool(row and row[2:4]==['COMPLETED','0:0'] and child.returncode==0 and not timed_out),
                 row=row,returncode=child.returncode,timed_out=timed_out,log_SHA256=hashlib.sha256(raw).hexdigest(),
                 network_lines=[x.rsplit('NET/',1)[-1].split(' ',1)[0] for x in raw.decode(errors='replace').splitlines() if 'NET/' in x],
                 log_tail=raw.decode(errors='replace').splitlines()[-5:] if child.returncode else [])
    save(phase+'-receipt.json',receipt)
    return receipt
pre=run('pre',35)
gpu=post=None
if pre['passed']:
    gpu=run('gpu',max(1,epoch+175-time.time()))
    if gpu['row']:
        post=run('post',35)
rank_rows=[]
for rank in [0,1]:
    path=W/'ranks'/f'{rank}.json'
    if path.is_file():rank_rows.append(json.loads(path.read_bytes()))
passed=bool(pre['passed'] and gpu and gpu['passed'] and post and post['passed'] and len(rank_rows)==2 and any(x.startswith('IB') and '/' in x for x in gpu['network_lines']) and not any(x.startswith('Socket/') for x in gpu['network_lines']))
traces={}
for path in sorted((W/'traces').glob('*.json')):
    raw=path.read_bytes()
    assert 0<len(raw)<65536
    traces[path.name]=dict(SHA256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
out=dict(traces=traces,request_id=REQUEST,claim=CLAIM,passed=passed,accounting=[r['row'] for r in [pre,gpu,post] if r and r['row']],
         accounting_columns=columns,pre=pre,gpu=gpu,post=post,ranks=rank_rows,wall_seconds=time.time()-epoch,
         rank_plan=[4096,65536,2097152],GPU_cards=2 if gpu else 0)
save('outcome.json',out)
print(json.dumps(out),flush=True)
