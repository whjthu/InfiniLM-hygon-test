"""Use the existing local A100 reference worktree, with finite owned cleanup."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


ROOT=Path(__file__).resolve().parent


def save(name,value):
    with (ROOT/name).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write('\n')


def main(epoch, req):
    deadline=time.monotonic()+760
    claim=json.loads((ROOT/'CLAIM.json').read_bytes())
    dep=json.loads((ROOT/'cuda-deps.json').read_bytes())
    assert req==claim['request'] and 0<=time.time()-epoch<15
    cpu=sorted(os.sched_getaffinity(0))[:8]
    assert cpu
    os.sched_setaffinity(0,cpu)
    pinned=claim['a100']
    uuid=pinned['uuid']
    def nvml():
        fields='index,uuid,pci.bus_id,memory.used'
        def read(args):
            return subprocess.check_output(['nvidia-smi',*args,'--format=csv,noheader,nounits'],
                                           text=True,timeout=8).splitlines()
        cards=[x.split(', ') for x in read(['--query-gpu='+fields])]
        apps=[x.split(', ') for x in read(['--query-compute-apps=pid,gpu_uuid,used_memory'])]
        selected,=[row for row in cards if row[1]==uuid]
        assert int(selected[0])==pinned['index'] and selected[2].upper()==pinned['pci']
        pids=[int(item[0]) for item in apps if len(item)==3 and item[1]==uuid]
        return dict(index=int(selected[0]),memory_used_MiB=int(selected[3]),compute_pids=pids)
    save('intent.json',dict(request=req,epoch=epoch,deadline=epoch+760))
    before=nvml()
    save('before.json',before)
    if before['compute_pids'] or before['memory_used_MiB']!=0:
        save('terminal.json',dict(request=req,gpu_passed=False,clean_post=False,error_class='FOREIGN_GPU_OCCUPANCY'))
        return 1
    old=Path(pinned['original_run'])
    environment=os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=uuid,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    bound=min(600,epoch+610-time.time(),deadline-time.monotonic()-80)
    assert bound>0
    cmd=[dep['python'],'-B','-u',str(ROOT/'run_platform.py'),'cuda',dep['reference'],
         str(old/'source/policy.json'),dep['weights'],str(old/'audit/replay/observed.pt')]
    forced=False
    failure=None
    with (ROOT/'gpu.log').open('xb') as f:
        child=subprocess.Popen(cmd,env=environment,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            ticks=int(Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19])
            save('owner.json',dict(pid=child.pid,pgid=child.pid,startticks=ticks))
            child.wait(timeout=bound)
        except BaseException as exc:
            failure=type(exc).__name__
        finally:
            if child.poll() is None:
                forced=True
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait(timeout=4)
    post=nvml()
    save('post.json',post)
    clean=not post['compute_pids'] and post['memory_used_MiB']==0
    done=bool(child.returncode==0 and not forced and not failure and clean and (ROOT/'result.json').is_file())
    result=dict(request=req,gpu_passed=done,clean_post=clean,rc=child.returncode,forced=forced,error_class=failure)
    save('terminal.json',result)
    print(json.dumps(result),flush=True)
    return int(not done)


if __name__=='__main__':
    import sys
    raise SystemExit(main(float(sys.argv[1]),sys.argv[2]))
