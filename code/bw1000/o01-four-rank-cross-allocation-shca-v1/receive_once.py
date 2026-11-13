import base64,hashlib,json,os,re,signal,socket,subprocess,sys,time
from pathlib import Path
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LABEL='o01-four-rank-cross-allocation-shca-v1'
W=R/'results'/LABEL
PLACEMENTS=[('1087656', 'f12r3n17'), ('1087656', 'f12r3n18'), ('1103606', 'f12r2n10'), ('1103606', 'f12r2n11')]
epoch=time.time()
anchor=time.monotonic()
def left(reserve=0):
    available=min(epoch+400-time.time(),400-(time.monotonic()-anchor))-reserve
    if available<=0:
        raise TimeoutError('original epoch exhausted')
    return available
def query(args):
    return subprocess.check_output(args,text=True,timeout=min(10,left())).strip()
columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList']
def job_name(phase,rank):
    return f'{LABEL}-{phase}-r{rank}'
def rows(parent,name):
    text=query(['sacct','-n','-P','-j',parent,'--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'])
    result=[line.split('|') for line in text.splitlines() if '|'+name+'|' in line]
    assert len(result)<=1
    return result[0] if result else None
def save(path,value):
    with (W/path).open('x') as file:
        json.dump(value,file,indent=2)
        file.flush()
        os.fsync(file.fileno())
assert not W.exists()
for rank,(parent,node) in enumerate(PLACEMENTS):
    info=query(['squeue','-h','-j',parent,'-o','%T|%D|%N']).split('|')
    assert info[:2]==['RUNNING','40']
    nodes=query(['scontrol','show','hostnames',info[2]]).splitlines()
    assert len(nodes)==len(set(nodes))==40 and node in nodes
    for line in query(['squeue','--steps','-h','-j',parent,'-o','%i|%N']).splitlines():
        handle,hosts=line.split('|',1)
        if handle not in {parent+'.batch',parent+'.extern'} and hosts not in ('','(null)'):
            assert node not in query(['scontrol','show','hostnames',hosts]).splitlines()
    assert all(rows(parent,job_name(phase,rank)) is None for phase in ('pre','gpu','post'))
W.mkdir(mode=0o700)
stat=Path('/proc/self/stat').read_text().split(') ',1)[1].split()
save('intent.json',dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=epoch+400,
     work_deadline=epoch+285,nodes=[node for _,node in PLACEMENTS],parents_by_node={n:p for p,n in PLACEMENTS},
     owner=dict(host=socket.gethostname(),PID=os.getpid(),startticks=int(stat[19])),
     source_files={n:x['SHA256'] for n,x in FILES.items()}))
for folder in ('src','pre','gpu-gate','post','ranks','traces','verbs-aliases'):
    (W/folder).mkdir()
for name,pin in FILES.items():
    assert Path(name).name==name
    raw=base64.b64decode(pin['b64'],validate=True)
    assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
    (W/'src'/name).write_bytes(raw)
for name in ('libibverbs.so','libibverbs.so.1'):
    (W/'verbs-aliases'/name).symlink_to('/rdma-provider-stack/libibverbs.so.1')
sys.path.insert(0,str(W/'src'))
from slurm_accounting import wait_terminal,is_terminal

def launch(phase,parent,node,rank):
    gpu=phase=='gpu'
    name=job_name(phase,rank)
    cmd=['srun','--jobid='+parent,'--job-name='+name,'--exact','--nodes=1','--ntasks=1',
         '--nodelist='+node,'--immediate=5','--cpus-per-task='+('4' if gpu else '1'),
         '--mem='+('8192M' if gpu else '1024M'),'--gres='+('dcu:1' if gpu else 'none'),
         '--time='+('3' if gpu else '1'),'/usr/local/python3.12/bin/python3','-B',
         str(W/'src/host_entry.py'),str(W),'gpu-gate' if gpu else phase]
    if gpu:
        shell='set -euo pipefail; source /opt/dtk/env.sh; export LD_LIBRARY_PATH=/candidate-opt/rccl-shca-net/lib:/candidate-opt/ucx-shca/lib:"$1/verbs-aliases":/rdma-provider-stack:$LD_LIBRARY_PATH RDMAV_DRIVERS=/rdma-provider-stack/libshca NCCL_NET=IBext_v8 NCCL_NET_PLUGIN=shca NCCL_IB_DISABLE=0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET UCX_MODULE_DIR=/candidate-opt/ucx-shca/lib/ucx UCX_NET_DEVICES=shca_0:1,shca_1:1,shca_2:1,shca_3:1 UCX_TLS=rc,sm,self OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1; exec python3 -B -u "$1/src/worker.py" "$1"'
        cmd+=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--rocm','--no-home',
              '--bind',str(W)+':'+str(W),'--bind',str(R/'build-deps/rdma-host-stack')+':/rdma-provider-stack:ro','--bind','/opt/hyhal:/opt/hyhal:ro',
              '--bind',str(R/'candidates/public-shca-layers-r2228/layer3-selected/opt')+':/candidate-opt:ro',
              str(R/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',shell,'collective',str(W)]
    log=W/(name+'.log')
    with log.open('xb') as stream:
        child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                               env=dict(os.environ,O01_GLOBAL_RANK=str(rank)))
    return parent,node,name,log,child

def run(phase):
    records=[]
    children=[]
    forced=False
    error=None
    try:
        for rank,(parent,node) in enumerate(PLACEMENTS):
            children.append(launch(phase,parent,node,rank))
        stop=time.monotonic()+min(left(50 if phase!='post' else 10),160 if phase=='gpu' else 30)
        for _,_,_,_,child in children:
            child.wait(timeout=max(.1,stop-time.monotonic()))
    except (OSError,subprocess.TimeoutExpired,TimeoutError) as failure:
        forced=True
        error=type(failure).__name__
    finally:
        for parent,node,name,log,child in children:
            if child.poll() is None:
                forced=True
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait(timeout=5)
            terminal=wait_terminal(lambda:rows(parent,name),time.monotonic()+min(15,left(2)),
                                   interval=.5,max_observations=30)
            row=terminal['row']
            raw=log.read_bytes()
            assert len(raw)<2**20
            records.append(dict(parent=parent,node=node,name=name,row=row,terminal_visible=terminal['terminal_visible'],
                returncode=child.returncode,log_SHA256=hashlib.sha256(raw).hexdigest(),
                network=re.findall(r'Using network\s+([\w.-]+)',raw.decode(errors='replace')),
                plugin_channels=len(re.findall(rb'via NET/IBext(?:_v8)?(?:/|\b)',raw)),
                socket_channels=len(re.findall(rb'via NET/Socket(?:/|\b)',raw))))
    passed=(not forced and len(records)==len(PLACEMENTS) and all(x['terminal_visible'] and x['row'][2:4]==['COMPLETED','0:0']
              and x['row'][5]==x['node'] and x['returncode']==0 for x in records))
    result=dict(passed=passed,steps=records,error=error,forced=forced)
    save(phase+'-receipt.json',result)
    return result

pre=run('pre')
gpu=post=None
if pre['passed']:
    gpu=run('gpu')
    if len(gpu['steps'])==len(PLACEMENTS) and all(x['terminal_visible'] for x in gpu['steps']):
        post=run('post')
rank_rows=[json.loads(p.read_bytes()) for p in sorted((W/'ranks').glob('*.json'))]
actual=list(pre['steps'])+(gpu or {}).get('steps',[])+(post or {}).get('steps',[])
passed=bool(gpu and post and gpu['passed'] and post['passed'] and {(row['rank'],row['node']) for row in rank_rows}==
    {(rank,node) for rank,(_,node) in enumerate(PLACEMENTS)} and len(rank_rows)==len(PLACEMENTS) and
    all(r['network']==['IBext_v8'] and r['plugin_channels']>0 and r['socket_channels']==0 for r in gpu['steps']))
out=dict(request_id=REQUEST,claim=CLAIM,passed=passed,accounting=[r['row'] for r in actual if r['row']],
         accounting_columns=columns,pre=pre,gpu=gpu,post=post,ranks=rank_rows,
         wall_seconds=time.monotonic()-anchor,GPU_cards=len(gpu['steps']) if gpu else 0)
save('outcome.json',out)
print(json.dumps(out),flush=True)
