"""Root-injected small package, one owned CPU process, no network/native/model child."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time

ROOT=Path('/home/whj/nvidia-baseline/alignment-assets')
NAME='training-USGS-T04-paired-training-views-CPU-v1'


def need(ok,why):
    if not ok:raise ValueError(why)

def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
def fds():
    found=[]
    for p in Path('/proc/self/fd').iterdir():
        try:target=os.readlink(p)
        except OSError:continue
        if target=='/dev/kfd' or target.startswith(('/dev/nvidia','/dev/dri/renderD')):found.append(int(p.name))
    return found

def main():
    allowed=sorted(set(os.sched_getaffinity(0))-{0,1});need(len(allowed)>=2,'Two allowed CPU cores excluding0/1 required');os.sched_setaffinity(0,set(allowed[:2]))
    need(os.sched_getaffinity(0)==set(allowed[:2]) and not fds(),'Actual CPU affinity/device FD admission differs')
    need(isinstance(INSTALL_PAYLOAD_B64,str) and len(INSTALL_PAYLOAD_B64)<1024**2,'Small root source-only payload required')
    payload=json.loads(base64.b64decode(INSTALL_PAYLOAD_B64,validate=True));epoch=payload['epoch']
    need(type(epoch) in [int,float] and math.isfinite(epoch) and 0<epoch<=time.time()<epoch+90,'Original120s epoch unavailable')
    deadline=epoch+120;work=epoch+100;final=epoch+110;mono=time.monotonic()+work-time.time()
    def check():
        if min(work-time.time(),mono-time.monotonic())<=0:raise TimeoutError('Original paired-view CPU work epoch exhausted')
    def alarm(*_):raise TimeoutError('Original paired-view CPU final alarm')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.001,final-time.time()))
    need(re.fullmatch('[0-9a-f]{32}',payload['request_id']) and re.fullmatch('[0-9a-f]{40}',payload['publication']),'Original request/publication required')
    package={name:base64.b64decode(body,validate=True) for name,body in payload['package'].items()}
    mf=package['SOURCE_FILES.json'];need(sha(mf)==payload['source_manifest_SHA256'],'Published source MF differs');pins=json.loads(mf)['files']
    need(set(package)==set(pins)|{'SOURCE_FILES.json'},'Exact package source set required')
    for name,raw in package.items():
        need(Path(name).name==name and name not in {'.','..'},'Flat source members only')
        if name!='SOURCE_FILES.json':need(len(raw)==pins[name]['bytes'] and sha(raw)==pins[name]['SHA256'],'Published member differs')
    claim=json.loads(package['CLAIM.json']);need(sha(package['CLAIM.json'])==payload['claim_SHA256'] and claim['root_GO'] is True and claim['CPU_GO'] is True
        and claim['training_GO'] is False and claim['GPU_GO'] is False,'Paired-view CPU scope not adopted')
    scope=ROOT/NAME;need(not scope.exists() and not scope.is_symlink(),'Original namespace exists; reconcile, no replay');scope.mkdir(mode=0o700)
    for name,raw in package.items():
        check();(scope/'src').mkdir(mode=0o700,exist_ok=True)
        with (scope/'src'/name).open('xb') as stream:stream.write(raw)
    fields=Path('/proc/self/stat').read_text().split(') ',1)[1].split()
    owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(fields[19]),uid=os.getuid(),host=os.uname().nodename,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    intent=dict(request_id=payload['request_id'],publication=payload['publication'],source_manifest_SHA256=sha(mf),claim_SHA256=payload['claim_SHA256'],epoch=epoch,deadline=deadline,work_deadline=work,final_deadline=final,
        owner=owner,CPU_affinity=allowed[:2],GPU=0,new_encodes=0,planned_arms=['domain','control'],planned_unique_rows_per_arm=45,planned_view_rows_per_arm=800)
    (scope/'intent.json').write_bytes(canonical(intent));result=dict(passed=False,**intent,arms={},actual_training_updates=0,training_GO=False,GPU_GO=False,Model_GO=False,caption_payload_exported=False)
    sys.path.insert(0,str(scope/'src'))
    try:
        from views_cpu import build_arm
        plan=json.loads(package['PLAN.json']);need(plan['namespace']==str(scope),'Original namespace differs')
        def progress(arm,details):
            with (scope/'progress.jsonl').open('ab') as stream:stream.write(canonical(dict(arm=arm,time=time.time(),**details)))
        for arm in ['domain','control']:
            check();result['arms'][arm]=build_arm(arm,plan['arms'][arm],scope/'src',scope/arm,check,progress)
            (scope/'partial-result.json').write_bytes(canonical(result))
        need(result['arms']['domain']['original_unique_index_sequence_SHA256']==result['arms']['control']['original_unique_index_sequence_SHA256'],'Two arms exposure schedule differs')
        result.update(passed=True,state='PAIRED_VIEWS800_ONLY_NO_NEW_ENCODE_TRAINING',new_encoded_rows=0,actual_new_view_rows=1600,actual_unique_source_rows=90,
            actual_GPU_rank_cursor_padding_or_sampler_qualification=False,CPU_reference_hashes_identity_candidates_only=True)
    except BaseException as error:
        result.update(state='KNOWN_FAILED_KEEP45_AND800_DENOMINATORS_NO_REPLAY',error_type=type(error).__name__,error='Paired view preparation failed; no caption body exported')
    finally:
        result.update(finished=time.time(),CPU_affinity_actual=sorted(os.sched_getaffinity(0)),self_device_FDs=fds(),peak_self_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            spawned_processes=0,owned_descendants_spawned=0,parent_must_observe_actual_process_exit=True,dev_or_blind_input_gold_read=False,new_network_calls=0)
        result['passed']=bool(result['passed'] and result['self_device_FDs']==[] and result['CPU_affinity_actual']==allowed[:2] and result['finished']<final)
        (scope/'paired-result.json').write_bytes(canonical(result));signal.setitimer(signal.ITIMER_REAL,0)
    # Strict metadata-only stdout; all source captions/800 views/occurrence ledgers stay in this remote private scope.
    print(json.dumps(result,sort_keys=True),flush=True);return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
