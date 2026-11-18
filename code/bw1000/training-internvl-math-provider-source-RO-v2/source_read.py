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

def initial_memory():
    fd=os.open('/proc/self/status',os.O_RDONLY)
    try:raw=os.read(fd,65537)
    finally:os.close(fd)
    need(len(raw)<=65536,'initial proc status bound')
    values={}
    for name in (b'VmSize:',b'VmRSS:'):
        lines=[line for line in raw.splitlines() if line.startswith(name)]
        need(len(lines)==1,'initial memory metric absent/duplicate')
        parts=lines[0].split();need(len(parts)==3 and parts[1].isdigit() and parts[2]==b'kB','initial memory metric not nonnegative kB')
        values[name[:-1].decode()]=int(parts[1])*1024
    return dict(VmSize_bytes=values['VmSize'],VmRSS_bytes=values['VmRSS'],
        status_bytes=len(raw),status_SHA256=hashlib.sha256(raw).hexdigest())

def read_once(context,claim):
    need(claim['root_GO'] is claim['READ_GO'] is True,'source read not adopted')
    endpoint=context['endpoint'];need(endpoint in SPECS and claim['endpoint_GO'][endpoint] is True,'this exact endpoint not adopted')
    epoch=context['epoch'];request=context['request_id'];publication=context['publication']
    need(type(epoch) in (int,float) and math.isfinite(epoch) and 0<=time.time()-epoch<35,'fresh single 40-second read epoch required')
    need(len(request)==32 and not set(request)-set('0123456789abcdef') and len(publication)==40
         and not set(publication)-set('0123456789abcdef'),'actual read request/publication required')
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,epoch+40-time.time())
    allowed=sorted(os.sched_getaffinity(0)-{0,1});need(len(allowed)>=2,'two allowed CPUs excluding0/1 required')
    os.sched_setaffinity(0,allowed[:2])
    memory=initial_memory();memory['AS_limit_bytes']=max(256*1024**2,memory['VmSize_bytes']+64*1024**2)
    memory['AS_is_RSS']=False;memory['headroom_is_design_not_measured_peak']=True
    memory_admitted=memory['VmSize_bytes']<=1024**3 and memory['AS_limit_bytes']<=2*1024**3
    memory['admitted']=memory_admitted
    if memory_admitted:resource.setrlimit(resource.RLIMIT_AS,(memory['AS_limit_bytes'],memory['AS_limit_bytes']))
    files=[];total=0;mono=time.monotonic()+epoch+40-time.time();memory_failed=False
    for spec in SPECS[endpoint]:
        need(time.monotonic()<mono,'same original read epoch exhausted')
        row=dict(name=spec['name'],declared_path=spec['path'],loaded_provider_identity_proven=False,attempted=False,actual_read_bytes=0,body_base64=None)
        if not memory_admitted or memory_failed:
            row.update(state='NOT_ATTEMPTED_MEMORY_ADMISSION' if not memory_admitted else 'NOT_ATTEMPTED_AFTER_MEMORY_ERROR')
            files.append(row);continue
        fd=None;chunks=[];raw=b'';chunk=b''
        try:
            row['attempted']=True
            path=Path(spec['path']);before=path.lstat()
            need(stat.S_ISREG(before.st_mode) and 0<before.st_size<=262144,'bounded regular source required; no final symlink')
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
            opened=os.fstat(fd);need(identity(before)==identity(opened),'source identity changed while opening')
            while row['actual_read_bytes']<before.st_size:
                need(time.monotonic()<mono,'same original source IO epoch exhausted')
                chunk=os.read(fd,min(65536,before.st_size-row['actual_read_bytes']))
                row['actual_read_bytes']+=len(chunk);total+=len(chunk)
                need(bool(chunk),'source truncated during bounded read')
                chunks.append(chunk)
            extra=os.read(fd,1);row['actual_read_bytes']+=len(extra);total+=len(extra)
            need(extra==b'','source grew beyond fixed stat size')
            raw=b''.join(chunks)
            need(len(raw)==before.st_size and len(raw)<=262144,'complete bounded source read differs')
            need(total<=524290,'two-source aggregate plus at most two growth probes cap')
            after_fd=os.fstat(fd);after_path=path.lstat()
            need(identity(before)==identity(after_fd)==identity(after_path),'source changed during passive read')
            digest=hashlib.sha256(raw).hexdigest()
            row.update(bytes=len(raw),SHA256=digest,stat_identity=identity(before))
            if endpoint=='a100_original':need(len(raw)==spec['bytes'] and digest==spec['SHA256'],'original A100 pinned source differs')
            row.update(state='COMPLETE_STABLE_SOURCE',bytes=len(raw),SHA256=digest,stat_identity=identity(before),
                body_base64=base64.b64encode(raw).decode('ascii'),source_kind='DECLARED_BASE_SOURCE' if endpoint=='hygon_declared_base' else 'PINNED_ORIGINAL_A100_SOURCE',
                actual_loaded_in_original_CPU357='NOT_PROVEN',stable_path_fd_identity=True)
        except MemoryError:
            # Release partial buffers before constructing only a tiny failure record.
            chunks.clear();raw=b'';chunk=b'';memory_failed=True
            row.update(state='MEMORY_ERROR_NO_BODY',error_type='MemoryError',body_base64=None)
        except (FileNotFoundError,PermissionError,OSError,ValueError) as error:
            reason=str(error)[:256]
            row.update(state='MISSING_OR_SOURCE_READ_FAILED',error_type=type(error).__name__,errno=getattr(error,'errno',None),
                private_reason=reason,reason_SHA256=hashlib.sha256(reason.encode()).hexdigest(),body_base64=None)
        finally:
            if fd is not None:os.close(fd)
            chunks.clear();raw=b'';chunk=b''
        files.append(row)
    completed=sum(r['state']=='COMPLETE_STABLE_SOURCE' for r in files)
    result=dict(state='EXACT_TWO_DECLARED_SOURCES_READ' if completed==2 else 'SOURCE_READ_PARTIAL_OR_FAILED_NO_FALLBACK',
        passed=completed==2,endpoint=endpoint,publication=publication,request_id=request,epoch=epoch,deadline=epoch+40,
        host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),UID=os.getuid(),CPU_affinity=allowed[:2],
        initial_memory=memory,memory_error_occurred=memory_failed,actual_attempted_files=sum(r['attempted'] for r in files),
        planned_files=2,completed_files=completed,missing_or_failed_files=2-completed,
        source_body_bytes=sum(r.get('bytes',0) for r in files if r['state']=='COMPLETE_STABLE_SOURCE'),actual_source_bytes_read=total,files=files,
        Torch_or_Apptainer_or_Slurm_or_GPU_or_weights_calls=0,source_modules_imported=False,
        original_CPU357_loaded_provider_proven=False,original_CPU357_failure_preserved=True,replay_allowed=False)
    need(len(json.dumps(result,sort_keys=True).encode())<=768*1024,'private source transport output cap')
    signal.setitimer(signal.ITIMER_REAL,0);return result

if __name__=='__main__':
    print(json.dumps(read_once(CONTEXT,READ_CLAIM),sort_keys=True))
