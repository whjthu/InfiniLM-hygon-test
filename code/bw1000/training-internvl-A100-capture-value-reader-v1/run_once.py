"""Bounded metadata writer and sequential CPU reader, for root dispatch after LLaVA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

D = Path(__file__).resolve().parent
A = Path('/home/whj/nvidia-baseline/alignment-assets')
LABEL = 'hygon-internvl-A100-capture-value-reader-v1'
H = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def save(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def rss(pid):
    try:
        rows = Path('/proc',str(pid),'status').read_text().splitlines()
        return int(next(r for r in rows if r.startswith('VmRSS:')).split()[1])*1024
    except (FileNotFoundError,StopIteration):
        return 0


def main(llava_outcome_sha, publication, request, manifest_sha):
    epoch,anchor = time.time(),time.monotonic()
    work_end,whole_end = anchor+780,anchor+900
    outdir = A/LABEL
    assert not outdir.exists(), 'original reader namespace exists; no replay'
    scope=json.loads((D/'SCOPE.json').read_bytes())
    assert scope['CPU_GO'] is True and scope['GPU_GO'] is False
    assert str(D)==scope['source_install'] and str(outdir)==scope['output_namespace']
    assert len(publication)==40 and all(c in '0123456789abcdef' for c in publication)
    assert len(request)==32 and all(c in '0123456789abcdef' for c in request)
    assert len(manifest_sha)==64 and H(D/'SOURCE_FILES.json')==manifest_sha
    assert len(llava_outcome_sha)==64 and all(c in '0123456789abcdef' for c in llava_outcome_sha)
    mf=json.loads((D/'SOURCE_FILES.json').read_bytes())
    for name,pin in mf['files'].items():
        assert H(D/name)==pin['SHA256'] and (D/name).stat().st_size==pin['bytes']
    # LLaVA completion is an explicit input, not elapsed time or inferred permission.
    llava=A/'hygon-llava-A100-first-updates-v1/outcome.json'
    assert H(llava)==llava_outcome_sha
    lv=json.loads(llava.read_bytes())
    assert lv['keeper_passed'] and lv['native_returncode']==0 and lv['cleanup']['owned_processes_remaining']==0
    q=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=12)
    assert q.returncode==0 and not q.stdout.strip(), 'A100 active workload; no reader IO overlap'
    source_pins=json.loads((D/'OBSERVER_SCHEMA_PINS.json').read_bytes())
    original=A/'hygon-internvl-A100-first-updates-v1'
    for name,pin in source_pins.items():assert H(original/'framework'/name)==pin
    assert H(original/'src/INTERNVL_NATIVE_BINDING.json')==H(D/'INTERNVL_NATIVE_BINDING.json')
    allowed=sorted(os.sched_getaffinity(0));assert len(allowed)>=2
    os.sched_setaffinity(0,allowed[:2])
    os.nice(10)
    outdir.mkdir(mode=0o700)
    save(outdir/'intent.json',dict(epoch=epoch,whole_seconds=900,work_seconds=780,publication=publication,request=request,
        source_manifest_SHA256=H(D/'SOURCE_FILES.json'),
        llava_outcome_SHA256=llava_outcome_sha,rank_denominator=8,file_denominator=200,update_denominator=16,
        CPU_affinity=allowed[:2],worker_RSS_limit_bytes=8*1024**3,maximum_active_workers=1,actual_GPU_cards=0,replay_allowed=False))
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',
               OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    rows=[];peak=0;failure=None
    for rank in range(8):
        if time.monotonic()>=work_end:
            failure='WORK_DEADLINE';break
        log=outdir/('rank-%02d.json'%rank);err=outdir/('rank-%02d.stderr'%rank)
        child=None;forced=None
        with log.open('xb') as stdout,err.open('xb') as stderr:
            child=subprocess.Popen(['/home/whj/nvidia-baseline/env-vlm/bin/python','-B','-u',str(D/'check_rank.py'),
                '--rank',str(rank),'--work-end',str(work_end)],stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env)
            # Popen owns an unreaped child; PID cannot be reused before wait.
            while child.poll() is None:
                current=rss(child.pid);peak=max(peak,current)
                if current>8*1024**3:forced='RSS_LIMIT'
                if time.monotonic()>=work_end:forced='WORK_DEADLINE'
                if log.stat().st_size>2*1024**2 or err.stat().st_size>2*1024**2:forced='METADATA_OUTPUT_LIMIT'
                if forced:
                    child.terminate()
                    try:child.wait(timeout=min(5,max(.1,whole_end-time.monotonic())))
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=min(5,max(.1,whole_end-time.monotonic())))
                    break
                time.sleep(.2)
        entry=dict(rank=rank,returncode=child.returncode,forced=forced,stdout_SHA256=H(log),stderr_SHA256=H(err),
                   stderr_bytes=err.stat().st_size,files_denominator=25,updates_denominator=2)
        try:
            assert log.stat().st_size<=2*1024**2
            entry['value']=json.loads(log.read_bytes())
        except (ValueError,AssertionError):entry['parse_failed']=True
        rows.append(entry)
        if forced or child.returncode or not entry.get('value',{}).get('passed'):
            failure=forced or 'RANK_READER_FAILED';break
    passed=len(rows)==8 and failure is None and all(r['value']['files_verified']==25 for r in rows)
    result=dict(passed=passed,publication=publication,request=request,rank_denominator=8,file_denominator=200,update_denominator=16,
        micro_forward_denominator=64,attempted_ranks=len(rows),completed_ranks=sum(bool(r.get('value',{}).get('passed')) for r in rows),
        completed_files=sum(r.get('value',{}).get('files_verified',0) for r in rows),failure=failure,
        observed_worker_peak_RSS_bytes=peak,RSS_sampling_seconds=.2,hard_memory_cgroup_enforced=False,
        workers=rows,elapsed_seconds=time.monotonic()-anchor,actual_GPU_cards=0,artifact_bytes_exported=0,
        tensor_finite_independently_checked=passed,full_200_SHA_checked=passed,
        full_optimizer_recompute=False,Hygon_alignment=False,formal_performance=False,formal_quality=False,replay_allowed=False)
    save(outdir/'outcome.json',result)
    print(json.dumps(dict(passed=passed,output_directory=str(outdir),outcome_SHA256=H(outdir/'outcome.json'),
        completed_ranks=result['completed_ranks'],completed_files=result['completed_files'],
        rank_denominator=8,file_denominator=200,update_denominator=16,failure=failure,
        elapsed_seconds=result['elapsed_seconds'],observed_worker_peak_RSS_bytes=peak,artifact_bytes_exported=0)))
    return 0 if passed else 1


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--llava-outcome-sha256',required=True)
    p.add_argument('--publication',required=True)
    p.add_argument('--request',required=True)
    p.add_argument('--source-manifest-sha256',required=True)
    args=p.parse_args()
    raise SystemExit(main(args.llava_outcome_sha256,args.publication,args.request,args.source_manifest_sha256))
