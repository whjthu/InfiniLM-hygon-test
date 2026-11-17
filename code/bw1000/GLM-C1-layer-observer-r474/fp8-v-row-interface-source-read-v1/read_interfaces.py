"""Root publishes these two sources then runs once through existing ordinary SSH.
Feed script on stdin to known HOST Python3.12 -I -S -B -u -, with arguments:
--scope-base64 B64 --scope-sha256 SHA --publication COMMIT --request ID --epoch E.
Root checks publication source/scope bytes before transport, starts E before that
check, and owns the exclusive local receipt. No remote file is written. Decode
returned bodies only into root's private local source directory after verification.
"""
import argparse,base64,hashlib,json,math,os,re,resource,signal,stat,time
from pathlib import Path
H=lambda b:hashlib.sha256(b).hexdigest()
ROOT='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
EXPECTED=(
 ('/incoming/GLM-M15-oproj-Model-r125/glm_fp8_runtime_oproj_r125.cpp','cdbfda3c1ebf919793ea38d590dad10dced8972e4703a032965a4a5a3242b6b2'),
 ('/incoming/GLM-M15-oproj-Model-r125/glm_fp8_cache_m15_r119.cpp','9a1eff3876a0271385be78be679e556e8ed2202dd582196ab0a495bb4a5fcb96'),
 ('/candidates/glm-deferred-attention-model-5f91ad7-r5096/src/glm_fp8_runtime_r809.hpp','94a13f0cb6378f226960bdf05a30ddb6cfbb430c8127a3ee126b8198582e02db'),
 ('/install/hygon-glm-5f91ad7-r3672/include/base/fp8_block_linear.h','42fdbf8fd12e44be2e6e49d5c743973877a786e34ac9e97b3e108825beb684db'))
def need(ok,code):
    if not ok:raise ValueError(code)
def identity(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def main(a):
    started=time.monotonic();available=a.epoch+60-time.time()
    q=dict(schema='GLM474-fp8-v-row-exact4-source-read-v1',request_id=a.request,publication=a.publication,epoch=a.epoch,deadline=a.epoch+60,scope_SHA256=a.scope_sha256,passed=False,planned_files=4,completed_files=0,attempted_files=0,total_read_bytes=0,files=[],phase='admission',replay_allowed=False,SDK_imports=0,Native_calls=0,GPU_calls=0,Slurm_calls=0,remote_writes=0)
    def left():
        remaining=min(a.epoch+60-time.time(),available-(time.monotonic()-started));need(math.isfinite(remaining) and remaining>0,'ORIGINAL60_EPOCH_EXHAUSTED');return remaining
    def alarm(*unused):raise TimeoutError('ORIGINAL60_EPOCH_EXHAUSTED')
    try:
        need(math.isfinite(a.epoch) and 0<available<=60,'ORIGINAL_EPOCH')
        signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,left())
        need(re.fullmatch('[0-9a-f]{40}',a.publication) and re.fullmatch('[0-9a-f]{32}',a.request) and re.fullmatch('[0-9a-f]{64}',a.scope_sha256),'IDENTITY_FORMAT')
        need(len(a.scope_base64)<=32768,'SCOPE_TRANSPORT_BOUND');raw=base64.b64decode(a.scope_base64,validate=True);need(H(raw)==a.scope_sha256,'SCOPE_SHA256');scope=json.loads(raw)
        need(scope['ROOT_GO'] is True and scope['remote_read_GO'] is True and all(scope[k] is False for k in ('compile_GO','Native_GO','GPU_GO','Model_GO')),'SOURCE_READ_NOT_ADOPTED')
        need(scope['max_invocations']==1 and scope['planned_files']==4 and scope['whole_epoch_seconds']==60 and scope['HOST_root']==ROOT,'EXACT_SCOPE')
        need(scope['CPU']==2 and scope['exclude_CPU_ids']==[0,1] and scope['RLIMIT_AS_bytes']==256*1024**2 and scope['aggregate_file_bytes_cap']==scope['individual_file_bytes_cap']==256*1024 and scope['stdout_bytes_cap']==384*1024,'EXACT_RESOURCE_BOUND')
        need(len(scope['files'])==4 and all(r['HOST_path']==ROOT+suffix and r['container_path']=='/refactor'+suffix and r['expected_SHA256']==sha for r,(suffix,sha) in zip(scope['files'],EXPECTED)),'EXACT_FOUR_PINS')
        allowed=sorted(set(os.sched_getaffinity(0))-{0,1});need(len(allowed)>=2,'TWO_CPU_REQUIRED');os.sched_setaffinity(0,set(allowed[:2]));need(set(os.sched_getaffinity(0))==set(allowed[:2]),'CPU_AFFINITY');q['CPU_affinity']=allowed[:2]
        resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2));q['RLIMIT_AS_bytes']=256*1024**2
        q['phase']='exact_metadata';prepared=[];total=0
        for suffix,sha in EXPECTED:
            left();logical=Path(ROOT+suffix);resolved=logical.resolve(strict=True)
            need(resolved.is_relative_to(Path(ROOT)) and resolved.name==logical.name,'SOURCE_RESOLVE_SCOPE')
            s=resolved.stat();need(stat.S_ISREG(s.st_mode) and 0<s.st_size<=256*1024,'SOURCE_FILE_BOUND');total+=s.st_size;need(total<=256*1024,'AGGREGATE_SOURCE_BOUND');prepared.append((logical,resolved,s,sha))
        q['phase']='stable_source_bytes'
        for logical,resolved,s,sha in prepared:
            left();q['attempted_files']+=1;q['current_source_path']=str(logical)
            with resolved.open('rb') as f:
                before=os.fstat(f.fileno());need(identity(s)==identity(before),'SOURCE_OPEN_CHANGED');raw=f.read(s.st_size);q['total_read_bytes']+=len(raw);after=os.fstat(f.fileno())
            need(len(raw)==s.st_size and identity(before)==identity(after)==identity(resolved.stat()) and logical.resolve(strict=True)==resolved,'SOURCE_STABLE_FD')
            actual=H(raw);q['current_observed_SHA256']=actual;need(actual==sha,'SOURCE_BODY_SHA256');raw.decode('utf8')
            q['files'].append(dict(path=str(logical),resolved_path=str(resolved),bytes=len(raw),SHA256=actual,body_base64=base64.b64encode(raw).decode('ascii')));q['completed_files']+=1
        left();q.update(passed=True,phase='complete',interface_support='Bodies returned only; kernel capability requires subsequent source review.')
    except BaseException as error:q.update(error_type=type(error).__name__,error_message_SHA256=H(str(error).encode()),phase_failed=q['phase'])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);q['elapsed_seconds']=time.monotonic()-started
        encoded=json.dumps(q,allow_nan=False,separators=(',',':')).encode();need(len(encoded)<=384*1024,'RETURN_BOUND');print(encoded.decode(),flush=True)
    return 0 if q['passed'] else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scope-base64',required=True);p.add_argument('--scope-sha256',required=True);p.add_argument('--publication',required=True);p.add_argument('--request',required=True);p.add_argument('--epoch',type=float,required=True);raise SystemExit(main(p.parse_args()))
