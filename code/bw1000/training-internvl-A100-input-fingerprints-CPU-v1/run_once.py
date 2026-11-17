"""One owned sequential CPU worker; client-origin deadline includes transport/install."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from common import D,P,Budget,require,sha,small,unique,save,owner


def rss(pid):
    try:
        raw=Path('/proc',str(pid),'status').read_bytes();require(len(raw)<=16384,'STATUS_BOUNDS')
        line=next((x for x in raw.decode().splitlines() if x.startswith('VmRSS:')),None)
        return int(line.split()[1])*1024 if line else 0
    except FileNotFoundError:return 0


def main(publication,request,epoch,manifest_sha):
    budget=Budget(epoch);out=Path(P['output_namespace']);child=None;owned=None;failure=None;peak=0;value=None;driver_queries=0
    require(str(D)==P['source_install'] and out.is_dir() and not out.is_symlink(),'EXACT_INSTALLED_NAMESPACE')
    def interrupted(*unused):raise TimeoutError('SUPERVISOR_DEADLINE_OR_SIGNAL')
    signal.signal(signal.SIGALRM,interrupted);signal.signal(signal.SIGTERM,interrupted)
    signal.setitimer(signal.ITIMER_REAL,budget.remaining(reserve=P['reserve_for_cleanup_seconds']))
    try:
        scope=unique((D/'CLAIM.json').read_bytes());require(scope['CPU_GO'] is True and scope['GPU_GO'] is False and scope['target_GO'] is False,'CLAIM_GO')
        mfraw,_=small(D/'SOURCE_FILES.json',131072);require(sha(mfraw)==manifest_sha,'SOURCE_MF')
        mf=unique(mfraw)
        for name,pin in mf['files'].items():small(D/name,256*1024,pin)
        allowed=sorted(os.sched_getaffinity(0));require(len(allowed)>=2,'CPU_AFFINITY');os.sched_setaffinity(0,allowed[:2]);os.nice(10)
        driver_queries+=1
        admission=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
            stdin=subprocess.DEVNULL,capture_output=True,timeout=min(12,budget.remaining(work=True,reserve=30)))
        save(out/'IO-admission.json',dict(returncode=admission.returncode,stdout_bytes=len(admission.stdout),stderr_bytes=len(admission.stderr),
            stdout_SHA256=sha(admission.stdout),stderr_SHA256=sha(admission.stderr),empty_compute_apps=not admission.stdout.strip()))
        require(admission.returncode==0 and not admission.stdout.strip(),'ACTIVE_GPU_WORKLOAD_NO_IO_OVERLAP')
        save(out/'intent.json',dict(publication=publication,request_id=request,epoch=epoch,deadline=epoch+P['whole_seconds'],
             work_deadline=epoch+P['work_offset_seconds'],owner=owner(os.getpid()),source_manifest_SHA256=manifest_sha,
             CPU_affinity=allowed[:2],files=64,max_workers=1,worker_RSS_sampled_limit_bytes=P['worker_RSS_sampled_limit_bytes'],
             hard_memory_cgroup_enforced=False,RLIMIT_AS_enforced=False,replay_allowed=False,device_compute_calls=0,driver_query_calls=driver_queries))
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        with (out/'worker.stdout').open('xb') as stdout,(out/'worker.stderr').open('xb') as stderr:
            budget.remaining(work=True,reserve=15)
            child=subprocess.Popen([P['python'],'-B','-u',str(D/'derive_inputs.py'),'--epoch',str(epoch),'--out',str(out)],
                stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env,cwd=out,start_new_session=True)
            owned=owner(child.pid);require(owned['PGID']==child.pid,'OWNED_SESSION');save(out/'worker-owner.json',owned)
            while child.poll() is None:
                budget.remaining(work=True);current=rss(child.pid);peak=max(peak,current)
                require(current<=P['worker_RSS_sampled_limit_bytes'],'SAMPLED_RSS_LIMIT')
                require((out/'worker.stdout').stat().st_size<=65536 and (out/'worker.stderr').stat().st_size<=65536,'WORKER_LOG_LIMIT')
                time.sleep(P['RSS_poll_seconds'])
        raw,_=small(out/'worker-result.json',16384);value=unique(raw)
        require(child.returncode==0 and value['passed'] is True and value['completed_files']==64,'WORKER_FAILED')
        small(out/'input-fingerprints.json',P['reference_limit_bytes'],value['output'])
    except BaseException as error:
        failure=type(error).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        cleanup_error=None
        try:
            if child is not None and child.poll() is None:
                # Only this unreaped Popen child can be signalled; no PID/PGID scan.
                current=owner(child.pid)
                require(owned and current['startticks']==owned['startticks'] and current['PID']==owned['PID'],'CHILD_IDENTITY_CHANGED')
                child.terminate()
                try:child.wait(timeout=min(5,budget.remaining(reserve=5)))
                except subprocess.TimeoutExpired:
                    child.kill();child.wait(timeout=min(5,budget.remaining(reserve=1)))
        except BaseException as error:
            cleanup_error=type(error).__name__;failure=failure or 'CLEANUP_NOT_CONFIRMED'
        clean=bool(child is None or child.poll() is not None)
        if value is None and (out/'worker-result.json').is_file():
            try:value=unique(small(out/'worker-result.json',16384)[0])
            except (ValueError,OSError):value=None
        passed=bool(failure is None and clean and child and child.returncode==0 and value and value.get('passed'))
        logs={}
        for name in ('worker.stdout','worker.stderr'):
            path=out/name
            if path.is_file():
                size=path.stat().st_size
                if size>65536:
                    logs[name]=dict(bytes=size,full_SHA256=None,over_limit=True)
                    failure=failure or 'WORKER_LOG_LIMIT';passed=False
                else:
                    raw,_=small(path,65536) if size else (b'',None)
                    logs[name]=dict(bytes=len(raw),SHA256=sha(raw))
        result=dict(passed=passed,request_id=request,publication=publication,epoch=epoch,deadline=epoch+P['whole_seconds'],
            source_manifest_SHA256=manifest_sha,worker=owned,worker_returncode=child.returncode if child else None,
            own_worker_reaped=clean,cleanup_error_type=cleanup_error,failure_type=failure,worker_result=value,completed_files=value.get('completed_files',0) if value else 0,
            file_denominator=64,observed_worker_peak_RSS_bytes=peak,RSS_sampling_seconds=P['RSS_poll_seconds'],
            hard_memory_cgroup_enforced=False,RLIMIT_AS_enforced=False,logs=logs,device_compute_calls=0,driver_query_calls=driver_queries,target_GO=False,
            other136_payloads_read=0,artifact_bytes_exported=0,RNG_equivalence=False,replay_allowed=False)
        identity=save(out/'outcome.json',result)
        print(json.dumps(dict(stage='CPU_DERIVATION_TERMINAL',request_id=request,passed=passed,outcome=identity,
            completed_files=result['completed_files'],file_denominator=64,own_worker_reaped=clean,
            observed_worker_peak_RSS_bytes=peak,failure_type=failure,device_compute_calls=0,driver_query_calls=driver_queries)),flush=True)
    return 0 if passed else 1
