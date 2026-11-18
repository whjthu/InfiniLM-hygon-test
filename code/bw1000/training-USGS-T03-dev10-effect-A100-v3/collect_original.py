"""Root injects CONTEXT, READ_DEADLINE and published READER_CLAIM; private stdout only."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import stat
import time

ROOT=Path('/home/whj/nvidia-baseline/alignment-assets/training-USGS-T03-dev10-effect-A100-v3')
ARMS=('initial','domain25','generic25')
H=lambda b:hashlib.sha256(b).hexdigest()


def need(ok,why):
    if not ok:raise ValueError(why)


def main():
    need(READER_CLAIM['READ_GO'] is True and READER_CLAIM['max_invocations']==1,'ROOT_READ_ADMISSION')
    need(CONTEXT['namespace']==str(ROOT) and isinstance(CONTEXT['epoch'],(int,float)) and CONTEXT['epoch']>0,'ACTUAL_CONTEXT_REQUIRED')
    need(all(isinstance(CONTEXT[k],str) and len(CONTEXT[k])==n and all(x in '0123456789abcdef' for x in CONTEXT[k]) for k,n in [('request_id',32),('publication',40),('source_manifest_SHA256',64),('claim_SHA256',64)]),'ACTUAL_CONTEXT_PINS')
    left=READ_DEADLINE-time.time();need(0<left<=60,'SINGLE_READ60_EPOCH')
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,left)
    cores=sorted(set(os.sched_getaffinity(0))-{0,1});need(len(cores)>=2,'TWO_CPU')
    os.sched_setaffinity(0,cores[:2]);os.nice(10)
    q=dict(context=CONTEXT,files={},original_execution_state='UNKNOWN',planned_generations=30,
           CPU_affinity=cores[:2],device_queries=0,Torch_imports=0,state_body_reads=0,
           gold_reads=0,remote_writes=0,replay_allowed=False,total_read_bytes=0,errors=[])

    def read(key,path,cap=262144):
        path=Path(path)
        try:
            need(not path.is_symlink(),'LEAF_SYMLINK')
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
            try:
                before=os.fstat(fd)
                need(stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid()
                     and 0<before.st_size<=cap and q['total_read_bytes']+before.st_size<=6*1024**2,'SMALL_JSON_BOUND')
                raw=b''
                while len(raw)<before.st_size:
                    block=os.read(fd,before.st_size-len(raw));need(block,'SHORT_FILE');raw+=block
                after=os.fstat(fd)
                sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
                need(sig(before)==sig(after)==sig(path.stat()),'ORIGINAL_CHANGED')
            finally:os.close(fd)
            q['total_read_bytes']+=len(raw)
            value=[json.loads(line) for line in raw.splitlines()] if path.suffix=='.jsonl' else json.loads(raw)
            q['files'][key]=dict(bytes=len(raw),SHA256=H(raw),value=value)
            return value
        except FileNotFoundError:
            q['files'][key]=dict(exists=False)
        except (OSError,ValueError,TypeError) as e:
            q['files'][key]=dict(exists=None,error_type=type(e).__name__)
        return None

    try:
        outcome=read('run/outcome.json',ROOT/'run/outcome.json')
        install=read('install-intent.json',ROOT/'install-intent.json',16384)
        read('install-failure.json',ROOT/'install-failure.json',16384)
        read('run/failure-private.json',ROOT/'run/failure-private.json',65536)
        owned=read('run/child-owner.json',ROOT/'run/child-owner.json',16384)
        read('run/after.json',ROOT/'run/after.json')
        control=read('run/control-state.json',ROOT/'run/control-state.json',16384)
        read('run/admission-query-failure-private.json',ROOT/'run/admission-query-failure-private.json',16384)
        read('run/post-query-failure-private.json',ROOT/'run/post-query-failure-private.json',16384)
        need(outcome is not None and install is not None,'ORIGINAL_TERMINAL_RECORD_MISSING')
        need(outcome['request_id']==install['request_id']==CONTEXT['request_id']
             and outcome['publication']==install['publication']==CONTEXT['publication']
             and outcome['epoch']==install['epoch']==CONTEXT['epoch']
             and outcome['deadline']==install['deadline']==CONTEXT['epoch']+900
             and outcome['finished_epoch']<=outcome['deadline'],'ORIGINAL_CONTEXT')
        need(outcome['source_manifest_SHA256']==install['source_manifest_SHA256']==CONTEXT['source_manifest_SHA256'],
             'ORIGINAL_PUBLISHED_SOURCE_BINDING')
        need(install['claim_SHA256']==CONTEXT['claim_SHA256'],'ORIGINAL_CLAIM')
        if owned is None and control is None:
            # Root's saved original client rc is a separate fact, never an invented PID anchor.
            if CONTEXT.get('original_client_returncode')==1 and CONTEXT.get('original_client_outcome')=='KNOWN_TERMINAL' and outcome['engineering_completed'] is False:
                q['original_execution_state']='KNOWN_TERMINAL_FAIL_OWNER_IDENTITY_UNAVAILABLE'
            q['original_process_identity']='UNKNOWN_NO_SAVED_CHILD_OR_SUPERVISOR_STARTTICKS'
            q['predictions_read']=False
            return q
        sup=owned['supervisor'] if owned is not None else control['supervisor']
        child=owned['child'] if owned is not None else None
        if child is None:
            need(control['request_id']==CONTEXT['request_id'] and control['publication']==CONTEXT['publication']
                 and control['epoch']==CONTEXT['epoch'] and control['source_manifest_SHA256']==CONTEXT['source_manifest_SHA256']
                 and control['exec_request_attempted'] is False and control['entry_released'] is False
                 and outcome['engineering_completed'] is False and outcome['native_released'] is False,'NO_CHILD_FAILED_ADMISSION')
        host=socket.gethostname();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        need(sup['host']==host and sup['boot_id']==boot and sup['scope_request_id']==CONTEXT['request_id']
             and sup['uid']==os.getuid(),'SAME_ORIGINAL_SUPERVISOR')
        if child is not None:
            need(child['host']==host and child['boot_id']==boot and child['request_id']==CONTEXT['request_id']
                 and child['UID']==os.getuid(),'SAME_ORIGINAL_CHILD')
        registry=read('run/owned-registry.json',ROOT/'run/owned-registry.json') if child is not None else {}
        need(isinstance(registry,dict) and len(registry)<=1024,'ORIGINAL_OWNED_REGISTRY')
        identities={int(x['PID']):int(x['startticks']) for x in registry.values()}
        identities.update({sup['PID']:sup['startticks']})
        if child is not None:identities.update({child['PID']:child['startticks']})
        observations=[]
        for pid,ticks in sorted(identities.items()):
            need(type(pid)is int and pid>0 and type(ticks)is int and ticks>0,'ORIGINAL_PID_SCHEMA')
            row=dict(PID=pid,startticks=ticks,state='UNKNOWN')
            try:
                p=Path('/proc')/str(pid)
                fields=(p/'stat').read_text().rsplit(')',1)[1].split()
                if int(fields[19])!=ticks:row['state']='ORIGINAL_IDENTITY_ABSENT_PID_REUSED'
                elif fields[0] in ('Z','X'):row['state']='EXITED_NOT_EXECUTING'
                else:row['state']='ORIGINAL_STILL_LIVE'
            except FileNotFoundError:row['state']='ABSENT'
            observations.append(row)
        q['original_process_observation']=dict(host=host,boot_id=boot,processes=observations)
        need(all(x['state'] in ('ABSENT','ORIGINAL_IDENTITY_ABSENT_PID_REUSED','EXITED_NOT_EXECUTING') for x in observations),
             'ORIGINAL_PROCESS_NOT_CONFIRMED_EXITED')
        q['original_execution_state']='TERMINAL_PASS' if outcome['engineering_completed'] else 'TERMINAL_FAIL'
        if child is None:
            q['failed_before_native_release']=True;q['predictions_read']=False
            return q
        for name in ('after.json','before.json','control-state.json','exec-intent.json','worker-admission.json',
                     'held-child-anchor.json','resource-samples.json'):
            if 'run/'+name not in q['files']:read('run/'+name,ROOT/'run'/name)
        mf=read('src/SOURCE_FILES.json',ROOT/'src/SOURCE_FILES.json')
        need(mf is not None and q['files']['src/SOURCE_FILES.json']['SHA256']==CONTEXT['source_manifest_SHA256'],
             'ACTUAL_MF')
        for name in ('CLAIM.json','INSTALL_PLAN.json','core/SOURCE_FILES.json','core/PLAN.json','core/CLAIM.json'):
            value=read('src/'+name,ROOT/'src'/name)
            need(value is not None and q['files']['src/'+name]['SHA256']==mf['files'][name]['SHA256']
                 and q['files']['src/'+name]['bytes']==mf['files'][name]['bytes'],'ACTUAL_SOURCE_MEMBER')
        plan=q['files']['src/core/PLAN.json']['value']
        native=plan['native_identities']
        need(native['path']=='/home/whj/nvidia-baseline/alignment-assets/training-USGS-T03-native-template-CPU-v2/run/native/native-identities-private.json'
             and native['bytes']==73407 and native['SHA256']=='34ce51a75a3ffa98ca9362ad1a0f339011e66927b9583c27df5f310a84e16b0e',
             'EXACT_DEV_NATIVE_REFERENCE')
        read('reference/native-identities',native['path'])
        for name in ('generation-plan.json','generation-finalized.json'):
            read('run/native/'+name,ROOT/'run/native'/name)
        for arm in ARMS:
            read('run/native/'+arm+'-predictions-private.jsonl',ROOT/'run/native'/(arm+'-predictions-private.jsonl'))
            for ordinal in range(10):
                name=arm+'-native-return-%02d.json'%ordinal
                read('run/native/'+name,ROOT/'run/native'/name,65536)
    except (OSError,ValueError,KeyError,TypeError,IndexError) as e:
        q['errors'].append(dict(boundary='terminal_source_or_metadata',error_type=type(e).__name__))
    return q


if __name__=='__main__':
    print(json.dumps(main(),separators=(',',':'),allow_nan=False))
