"""Two passive snapshots of the original own loader PID; no device imports."""
from pathlib import Path
import hashlib,json,os,socket,time,traceback
NODES=['a14r2n08','a14r2n09','a14r2n10','a14r2n12']
RUN='tp32-full-cohort-20261105-c'
MF='1a817f0a57fba50d8f68287e1ca2438293e8965036f63e72027845e1d1fbbd4a'
EPOCH=1793870246.1336355
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
FOLDER=R/'results/Kimi-r461-full-model-tp32-full-cohort-20261105-c'
OUT=R/'results/Kimi637-loader-passive-probe-v1'
def stat(path):
    raw=path.read_text();fields=raw.rsplit(')',1)[1].split()
    return dict(state=fields[0],PPID=int(fields[1]),PGID=int(fields[2]),minor_faults=int(fields[7]),major_faults=int(fields[9]),user_ticks=int(fields[11]),system_ticks=int(fields[12]),threads=int(fields[17]),startticks=int(fields[19]))
def report(node):
    path=FOLDER/('node%d.json'%node)
    with path.open('rb')as f:raw=f.read(4*1024**2+1)
    assert len(raw)<=4*1024**2,'Bounded report size exceeded'
    v=json.loads(raw)
    assert(v['node'],v['hostname'],v['run'],v['source_manifest_SHA256'],v['epoch'],v['actual_step'],v['UID'])==(node,NODES[node],RUN,MF,EPOCH,'910380.637',154459)
    return v,dict(path=str(path),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),completed=v['completed'],passed=v['passed'],error=v.get('error'),Model_READY=v.get('Model_READY'),fullModel_forward=v['fullModel_forward'],load_progress=[e for e in v['events']if e['event']=='load_progress'][-1:],last_event=v['events'][-1:]if v['events']else[])
def sample(node,binding=None):
    v,pin=report(node);pid=v['PID'];p=Path('/proc')/str(pid)
    status=(p/'status').read_text();uids=[int(x)for x in next(line for line in status.splitlines()if line.startswith('Uid:')).split()[1:]]
    assert uids==[154459]*4 and os.getuid()==154459
    raw=(p/'cmdline').read_bytes();argv=raw.rstrip(b'\0').split(b'\0')
    assert b'/refactor/incoming/Kimi-vocab-parallel-embedding-r461/full-model-v3/node.py'in argv and b'--execute'in argv
    before=stat(p/'stat');assert before['PGID']==v['PGID']
    boot=int(next(line for line in Path('/proc/stat').read_text().splitlines()if line.startswith('btime ')).split()[1])
    start_unix=boot+before['startticks']/os.sysconf('SC_CLK_TCK')
    assert EPOCH-2<=start_unix<=v['events'][0]['unix']+2,'PID start must belong to this GPU epoch before original first event'
    identity=dict(PID=pid,UID=154459,startticks=before['startticks'],PGID=before['PGID'],step='910380.637',run=RUN,source=MF,epoch=EPOCH,cmdline_SHA256=hashlib.sha256(raw).hexdigest())
    if binding is not None:assert identity==binding,'Original PID/startticks/identity changed'
    io={}
    try:
        for line in(p/'io').read_text().splitlines():k,value=line.split(':',1);io[k]=int(value)
    except OSError as error:io=dict(error=repr(error))
    tids=sorted(int(x.name)for x in(p/'task').iterdir()if x.name.isdigit());assert len(tids)<=256,'Bounded thread inventory exceeded'
    threads=[]
    for tid in tids:
        t=p/'task'/str(tid)
        try:
            row=dict(TID=tid,**stat(t/'stat'))
            try:row['wchan']=(t/'wchan').read_text()[:128].strip()
            except OSError as error:row['wchan_error']=repr(error)
            threads.append(row)
        except OSError as error:threads.append(dict(TID=tid,error=repr(error)))
    after=stat(p/'stat');assert after['startticks']==before['startticks']
    return dict(unix=time.time(),monotonic=time.monotonic(),identity=identity,PID_start_unix_estimate=start_unix,start_clock_resolution='boot btime integer seconds plus proc ticks; +/-2s identity bound only',report=pin,process_before=before,process_after=after,io=io,threads=threads)
def main():
    node=NODES.index(socket.gethostname().split('.')[0]);assert os.environ['SLURM_JOB_ID']=='910380'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==os.environ.get('HIP_VISIBLE_DEVICES')==os.environ.get('ROCR_VISIBLE_DEVICES')=='-1'
    OUT.mkdir(exist_ok=True);dest=OUT/('node%d.json'%node);assert not dest.exists(),'One original probe only'
    result=dict(node=node,host=NODES[node],actual_CPU_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],passed=False,samples=[],GPU=0,Native_calls=0,signals=0,attach=False,weight_or_memory_read=False)
    try:
        result['clock_ticks_per_second']=os.sysconf('SC_CLK_TCK')
        first=sample(node);result['samples'].append(first)
        with dest.open('x')as f:json.dump(result,f,indent=2)
        time.sleep(10)
        second=sample(node,first['identity']);result['samples'].append(second)
        dt=second['monotonic']-first['monotonic'];assert 9<=dt<=25
        ticks=result['clock_ticks_per_second'];delta={k:second['process_after'][k]-first['process_after'][k]for k in ['user_ticks','system_ticks','minor_faults','major_faults']};delta['wall_s']=dt
        delta['user_seconds']=delta['user_ticks']/ticks;delta['system_seconds']=delta['system_ticks']/ticks
        if 'error'not in first['io']and'error'not in second['io']:delta['io']={k:second['io'][k]-first['io'][k]for k in first['io']}
        result.update(passed=True,delta=delta,limits='CPU ticks/io counters and wchan are observations, not exclusive causes; mmap/page-cache traffic and GPU busy are not inferred from read_bytes alone. wchan0/permission errors are unknown.')
    except BaseException as error:result.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        result['completed']=True;temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(dest)
    print(json.dumps(dict(node=node,path=str(dest),bytes=dest.stat().st_size,SHA256=hashlib.sha256(dest.read_bytes()).hexdigest(),passed=result['passed'],delta=result.get('delta'),error=result.get('error'))))
if __name__=='__main__':main()
