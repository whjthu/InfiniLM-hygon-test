"""Dispatch one two-node collective with owned steps and post-device checks."""
import base64,hashlib,json,os,re,signal,socket,subprocess,sys,time
from pathlib import Path
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
PARENT='1103606'
NODES=['f13r1n08', 'f13r1n09']
LABEL='GLM474-TP16-transport-pair-v1'
W=R/'results'/LABEL
MODES=('socket','shca','shca','socket','socket','shca')
ARMS=tuple(f'arm-{i}' for i in range(len(MODES)))
epoch=time.time()
anchor=time.monotonic()
columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList']
def left():
    return min(epoch+800-time.time(),800-(time.monotonic()-anchor))
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
assert not W.exists() and all(not rows(LABEL+'-'+phase) for phase in ('pre',*ARMS,'post'))
parent=query(['squeue','-h','-j',PARENT,'-o','%T|%D|%N']).split('|')
assert parent[:2]==['RUNNING','40']
assert set(NODES)<=set(query(['scontrol','show','hostnames',parent[2]]).splitlines())
def verify_nodes_free():
    for line in query(['scontrol','show','step',PARENT,'--oneliner']).splitlines():
        fields=dict(part.split('=',1) for part in line.split() if '=' in part)
        step=fields['StepId']
        if step.endswith('.extern'):
            continue
        assert not set(NODES).intersection(query(['scontrol','show','hostnames',fields['NodeList']]).splitlines())
verify_nodes_free()
W.mkdir(mode=0o700)
stat=Path('/proc/self/stat').read_text().split(') ',1)[1].split()
save('intent.json',dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=epoch+800,work_deadline=epoch+705,
    nodes=NODES,owner=dict(host=socket.gethostname(),PID=os.getpid(),startticks=int(stat[19])),
    source_files={n:x['SHA256'] for n,x in FILES.items()}))
for folder in ('src','pre','post',*ARMS):
    (W/folder).mkdir()
for name,pin in FILES.items():
    assert Path(name).name==name
    raw=base64.b64decode(pin['b64'],validate=True)
    assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
    (W/'src'/name).write_bytes(raw)
aliases=W/'verbs-aliases'
aliases.mkdir()
for name in ('libibverbs.so','libibverbs.so.1'):
    (aliases/name).symlink_to('/rdma-provider-stack/libibverbs.so.1')
sys.path.insert(0,str(W/'src'))
from verify_transport import network_channels
from terminal_receipts import collect_receipts
from slurm_accounting import wait_terminal
def run(phase,timeout):
    gpu=phase in ARMS
    label=LABEL+'-'+phase
    cmd=['srun','--jobid='+PARENT,'--job-name='+label,'--exact','--nodes=2','--ntasks='+('16' if gpu else '2'),'--ntasks-per-node='+('8' if gpu else '1'),
         '--distribution=block:block','--nodelist='+','.join(NODES),'--immediate=5','--kill-on-bad-exit=1',
         '--cpus-per-task='+('4' if gpu else '1'),'--mem='+('16384M' if gpu else '1024M'),
         '--gres='+('dcu:8' if gpu else 'none'),'--time='+('3' if gpu else '1'),
         '/usr/local/python3.12/bin/python3','-B',str(W/'src/host_entry.py'),str(W),'gpu-gate' if gpu else phase]
    if gpu:
        cmd=cmd[:cmd.index('/usr/local/python3.12/bin/python3')]
        scope='/refactor/results/'+LABEL
        mode=MODES[ARMS.index(phase)]
        shell='''set -euo pipefail
source "$1/src/GPU_ENV.sh"
export LD_LIBRARY_PATH=/candidate-opt/rccl-shca-net/lib:/candidate-opt/ucx-shca/lib:"$1/verbs-aliases":/rdma-provider-stack:$LD_LIBRARY_PATH
export RDMAV_DRIVERS=/rdma-provider-stack/libshca UCX_MODULE_DIR=/candidate-opt/ucx-shca/lib/ucx
export UCX_NET_DEVICES=shca_0:1,shca_1:1,shca_2:1,shca_3:1 UCX_TLS=rc,sm,self
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
if [ "$3" = shca ]; then
 export NCCL_NET=IBext_v8 NCCL_NET_PLUGIN=shca NCCL_IB_DISABLE=0
else
 export NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=1
fi
exec /usr/bin/python3 -B -u "$1/src/worker.py" "$2" "$3"
'''
        cmd+=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--rocm','--no-home','--pwd','/',
              '--bind',str(R)+':/refactor:ro','--bind',str(W)+':'+scope+':rw',
              '--bind',str(R/'build-deps/rdma-host-stack')+':/rdma-provider-stack:ro',
              '--bind',str(R/'candidates/public-shca-layers-r2228/layer3-selected/opt')+':/candidate-opt:ro',
              '--bind','/opt/hyhal:/opt/hyhal:ro',str(R/'containers/dtk26-sandbox-r1'),
              '/bin/bash','--noprofile','--norc','-c',shell,'component',scope,phase,mode]
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
    if gpu and receipt['passed']:
        try:
            receipt['network']=network_channels(raw.decode(errors='replace'),mode)
        except ValueError as error:
            receipt.update(passed=False, network_error=str(error))
    save(phase+'-receipt.json',receipt)
    return receipt
pre=run('pre',35)
arms={}
post=None
error=None
attempted=False
unknown=False
try:
    if pre['passed']:
        for arm in ARMS:
            verify_nodes_free()
            attempted=True
            unknown=True
            arms[arm]=run(arm,75)
            unknown=not arms[arm]['terminal_visible']
            if not arms[arm]['passed']:
                break
except Exception as failure:
    error=type(failure).__name__
finally:
    if attempted and not unknown:
        try:
            post=run('post',35)
        except Exception as failure:
            error=error or 'post:'+type(failure).__name__
passed=bool(pre['passed'] and post and post['passed'] and len(arms)==len(ARMS)
            and all(r['passed'] for r in arms.values()))
rank_data={}
if passed:
    names=[f'{arm}/{rank:02d}.json' for arm in ARMS for rank in range(16)]
    collected=collect_receipts(W,names,time.monotonic()+max(0,min(18,left()-5)),max_attempts=36)
    passed=not collected['missing']
    save('rank-visibility.json',dict(missing=collected['missing'],observations=collected['observations']))
    if passed:
        from statistics import median
        for arm,mode in zip(ARMS,MODES):
            receipts=[collected['receipts'][f'{arm}/{rank:02d}.json'] for rank in range(16)]
            rr=[receipt['value'] for receipt in receipts]
            assert len({r['uid_SHA256'] for r in rr})==1
            assert all(r['mode']==mode and r['step']==arms[arm]['row'][0]
                       and r['rank']==rank and r['node']==NODES[rank//8] and r['comm_destroyed']
                       and r['min_free_bytes']>=8589934592 for rank,r in enumerate(rr))
            for node in NODES:
                rows_for_node=[r for r in rr if r['node']==node]
                before=json.loads((W/'pre'/f'{node}.json').read_bytes())
                assert sorted(r['pci'] for r in rows_for_node)==before['pci']
                assert len({r['kfd_id'] for r in rows_for_node})==8
            checks=[]
            for index,(elements,operation) in enumerate([(1024,'allreduce'),(1024,'allgather'),
                                                         (6144,'allreduce'),(6144,'allgather')]):
                values=[r['checks'][index] for r in rr]
                assert all(v['elements']==elements and v['operation']==operation and v['checked']==25
                           and len(v['latency_ns'])==20 and all(type(ns) is int and ns>0 for ns in v['latency_ns'])
                           for v in values)
                assert len({v['output_SHA256'] for v in values})==1
                checks.append(dict(elements=elements,operation=operation,output_SHA256=values[0]['output_SHA256'],
                                   median_slowest_rank_ns=median(max(sample) for sample in
                                       zip(*(v['latency_ns'] for v in values)))))
            rank_data[arm]=dict(mode=mode,checks=checks,ranks=16,rank_SHA256=[r['sha256'] for r in receipts])
if passed:
    checks_by_arm=[[x['output_SHA256'] for x in r['checks']] for r in rank_data.values()]
    passed=all(checks==checks_by_arm[0] for checks in checks_by_arm)
out=dict(request_id=REQUEST,claim=CLAIM,passed=passed,error=error,order=MODES,pre=pre,arms=arms,post=post,
         rank_data=rank_data,accounting_columns=columns,wall_seconds=time.monotonic()-anchor,
         component_only=True)
save('outcome.json',out)
print(json.dumps(out),flush=True)
