"""One passive read of exactly two declared source files; never import their modules."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import stat
import time

SANDBOX='/public/home/aclsylqidf/hygon-inference-refactor-20261014/containers/dtk26-sandbox-r1'
SPECS={
    'hygon_declared_base':[
        dict(name='modeling_internvl.py',path=SANDBOX+'/usr/local/lib/python3.10/dist-packages/transformers/models/internvl/modeling_internvl.py'),
        dict(name='modeling_qwen3.py',path=SANDBOX+'/usr/local/lib/python3.10/dist-packages/transformers/models/qwen3/modeling_qwen3.py')],
    'a100_original':[
        dict(name='modeling_internvl.py',path='/home/whj/nvidia-baseline/env-vlm/lib/python3.12/site-packages/transformers/models/internvl/modeling_internvl.py',
             bytes=37163,SHA256='fec9f18a83dea42dd32730555cac5a66ece67ef5f236f32967fed1f296d42bd6'),
        dict(name='modeling_qwen3.py',path='/home/whj/nvidia-baseline/env-vlm/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py',
             bytes=23353,SHA256='fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab')]
}

def need(ok,why):
    if not ok:raise ValueError(why)

def identity(row):
    return {k:getattr(row,k) for k in ('st_dev','st_ino','st_size','st_mode','st_uid','st_gid','st_mtime_ns','st_ctime_ns')}

def read_once(context,claim):
    need(claim['root_GO'] is claim['READ_GO'] is True,'source read not adopted')
    endpoint=context['endpoint'];need(endpoint in SPECS and claim['endpoint_GO'][endpoint] is True,'this exact endpoint not adopted')
    epoch=context['epoch'];request=context['request_id'];publication=context['publication']
    need(type(epoch) in (int,float) and math.isfinite(epoch) and 0<=time.time()-epoch<35,'fresh single 40-second read epoch required')
    need(len(request)==32 and not set(request)-set('0123456789abcdef') and len(publication)==40
         and not set(publication)-set('0123456789abcdef'),'actual read request/publication required')
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,epoch+40-time.time())
    allowed=sorted(os.sched_getaffinity(0)-{0,1});need(len(allowed)>=2,'two allowed CPUs excluding0/1 required')
    os.sched_setaffinity(0,allowed[:2]);resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2))
    files=[];total=0;mono=time.monotonic()+epoch+40-time.time()
    for spec in SPECS[endpoint]:
        need(time.monotonic()<mono,'same original read epoch exhausted')
        row=dict(name=spec['name'],declared_path=spec['path'],loaded_provider_identity_proven=False)
        fd=None
        try:
            path=Path(spec['path']);before=path.lstat()
            need(stat.S_ISREG(before.st_mode) and 0<before.st_size<=262144,'bounded regular source required; no final symlink')
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
            opened=os.fstat(fd);need(identity(before)==identity(opened),'source identity changed while opening')
            with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(262145)
            need(len(raw)==before.st_size and len(raw)<=262144,'complete bounded source read differs')
            total+=len(raw);need(total<=524288,'two-source aggregate cap')
            after_fd=os.fstat(fd);after_path=path.lstat()
            need(identity(before)==identity(after_fd)==identity(after_path),'source changed during passive read')
            digest=hashlib.sha256(raw).hexdigest()
            row.update(bytes=len(raw),SHA256=digest,stat_identity=identity(before))
            if endpoint=='a100_original':need(len(raw)==spec['bytes'] and digest==spec['SHA256'],'original A100 pinned source differs')
            row.update(state='COMPLETE_STABLE_SOURCE',bytes=len(raw),SHA256=digest,stat_identity=identity(before),
                body_base64=base64.b64encode(raw).decode('ascii'),source_kind='DECLARED_BASE_SOURCE' if endpoint=='hygon_declared_base' else 'PINNED_ORIGINAL_A100_SOURCE',
                actual_loaded_in_original_CPU357='NOT_PROVEN',stable_path_fd_identity=True)
        except (FileNotFoundError,PermissionError,OSError,ValueError) as error:
            reason=str(error)[:256]
            row.update(state='MISSING_OR_SOURCE_READ_FAILED',error_type=type(error).__name__,errno=getattr(error,'errno',None),
                private_reason=reason,reason_SHA256=hashlib.sha256(reason.encode()).hexdigest(),body_base64=None)
        finally:
            if fd is not None:os.close(fd)
        files.append(row)
    completed=sum(r['state']=='COMPLETE_STABLE_SOURCE' for r in files)
    result=dict(state='EXACT_TWO_DECLARED_SOURCES_READ' if completed==2 else 'SOURCE_READ_PARTIAL_OR_FAILED_NO_FALLBACK',
        passed=completed==2,endpoint=endpoint,publication=publication,request_id=request,epoch=epoch,deadline=epoch+40,
        host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),UID=os.getuid(),CPU_affinity=allowed[:2],
        planned_files=2,completed_files=completed,missing_or_failed_files=2-completed,
        source_body_bytes=sum(r.get('bytes',0) for r in files if r['state']=='COMPLETE_STABLE_SOURCE'),actual_source_bytes_read=total,files=files,
        Torch_or_Apptainer_or_Slurm_or_GPU_or_weights_calls=0,source_modules_imported=False,
        original_CPU357_loaded_provider_proven=False,original_CPU357_failure_preserved=True,replay_allowed=False)
    need(len(json.dumps(result,sort_keys=True).encode())<=768*1024,'private source transport output cap')
    signal.setitimer(signal.ITIMER_REAL,0);return result

if __name__=='__main__':
    print(json.dumps(read_once(CONTEXT,READ_CLAIM),sort_keys=True))
