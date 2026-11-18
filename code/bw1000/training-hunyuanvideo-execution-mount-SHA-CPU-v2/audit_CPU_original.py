"""Independent small-receipt/accounting read. Root injects CONTEXT/READ_DEADLINE after actual CPU terminal."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import sys
SOURCE=Path(__file__).resolve().parent
if str(SOURCE) not in sys.path:sys.path.insert(0,str(SOURCE))
from pending_placeholder import versions, classify

ROOT=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-hunyuanvideo-execution-mount-SHA-CPU-v2')
COLS=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList','AllocTRES']


def need(ok,reason):
    if not ok:raise ValueError(reason)


def small(path,cap=65536):
    for item in [*reversed(path.parents),path]:need(not item.is_symlink(),'ORIGINAL_RECORD_SYMLINK')
    before=path.stat();need(path.is_file() and before.st_uid==os.getuid()==154459 and 0<before.st_size<=cap,'ORIGINAL_SMALL_BOUND')
    with path.open('rb',buffering=0) as stream:
        raw=stream.read(cap+1);after=os.fstat(stream.fileno())
    key=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(key(before)==key(after)==key(path.stat()) and len(raw)==before.st_size,'ORIGINAL_RECORD_CHANGED')
    return json.loads(raw),dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())


def main():
    left=READ_DEADLINE-time.time();need(0<left<=60,'INDEPENDENT_READ_BUDGET')
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,left)
    outcome,outpin=small(ROOT/'outcome.json');worker,workerpin=small(ROOT/'weight-receipt.json')
    intent,intentpin=small(ROOT/'intent.json')
    for key in ('request_id','publication','epoch','source_manifest_SHA256'):
        need(outcome[key]==worker[key]==intent[key]==CONTEXT[key],'EXACT_ORIGINAL_CPU_REQUEST')
    version=intent['current_version_evidence']
    need(version==versions(version['original_scontrol'],version['original_srun'],version['original_server_version_lines']),
         'ACTUAL_CLIENT_SERVER_VERSION_GATE')
    need(intent['original_TDB_request_state']=='UNKNOWN_NO_CANCEL_NO_RESERVATION_RELEASE','KEEP_ORIGINAL_REQUEST_UNKNOWN')
    for item in intent['original_step_observations']:
        kind=classify(item['original_fields'],intent['expanded_parent_hosts'],item['expanded_step_hosts'],154459)
        need(kind==item['classification'] and item['original_request_state']=='UNKNOWN_NOT_INFERRED_FROM_STEP_CLASSIFICATION',
             'ACTUAL_ORIGINAL_STEP_CLASSIFICATION')
        if kind=='NUMBERED_STEP_REQUIRES_NODE_CONFLICT_CHECK':
            need('f13r3n13' not in item['expanded_step_hosts'],'ACTUAL_NUMBERED_STEP_NODE_CONFLICT')
    need(outcome['passed'] is True and worker['passed'] is True and outcome['current_step_outcome']=='TERMINAL'
         and worker['deadline']==worker['epoch']+900 and worker['finished_epoch']<=worker['work_deadline']<=worker['epoch']+780,
         'ORIGINAL_CPU_PASS_REQUIRED')
    mf,mfpin=small(ROOT/'source/SOURCE_FILES.json',262144)
    need(mfpin['SHA256']==CONTEXT['source_manifest_SHA256'],'ORIGINAL_PUBLISHED_CPU_SOURCE')
    for name,pin in mf['files'].items():
        path=ROOT/'source'/name;need('..' not in Path(name).parts and not path.is_symlink(),'EXACT_SOURCE_MEMBER_PATH')
        raw=path.read_bytes();need(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'],'ORIGINAL_SOURCE_CLOSURE')
    step=worker['CPU_step'];need(re.fullmatch(r'1103606\.[0-9]+',step),'EXACT_KNOWN_STEP')
    lines=subprocess.check_output(['sacct','-n','-P','-j',step,'-o','JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'],
        text=True,timeout=min(12,READ_DEADLINE-time.time())).splitlines()
    rows=[line.split('|') for line in lines if line.split('|')[0]==step]
    need(len(rows)==1 and len(rows[0])==7,'EXACT_ACCOUNTING_UNIQUE')
    row=rows[0];tres=dict(x.split('=',1) for x in row[6].split(',') if '=' in x)
    need(row==outcome['actual_accounting'] and row[1]=='training-hunyuanvideo-execution-mount-SHA-CPU-v2'
         and row[2:4]==['COMPLETED','0:0'] and row[5]==worker['node']=='f13r3n13'
         and tres.get('cpu')=='2' and tres.get('mem') in ('4G','4096M')
         and not any(k.startswith('gres/') and v!='0' for k,v in tres.items()),'ACTUAL_C0_2CPU4G_0GPU')
    need(outcome['held_child']['reaped'] is True and worker['payload_bytes_read']==33306632192
         and worker['weight_SHA256']=='71f9affa1115fef2b14bd41fba30eab966fe80c9ed98e0fcba495dbc6d8fff86'
         and worker['tensor_decodes']==worker['Torch_SDK_imports']==worker['GPU_cards']==0
         and worker['original_replay_inputs_read'] is False,'ACTUAL_FULL_SHA_AND_NO_MODEL')
    authority=dict(schema='T09_ROOT_INDEPENDENT_CPU_EXECUTION_AUTHORITY_V1',passed=True,
        root_independent_terminal_verified=True,worker_exit_evidence='EXACT_SACCT_COMPLETED0_FOREGROUND_NO_CHILDREN',
        original_request_id=CONTEXT['request_id'],CPU_source_manifest_SHA256=CONTEXT['source_manifest_SHA256'],
        CPU_publication=CONTEXT['publication'],CPU_epoch=CONTEXT['epoch'],accounting_columns=COLS,actual_accounting=row,
        worker_receipt=dict(relative_path='training-hunyuanvideo-execution-mount-SHA-CPU-v2/weight-receipt.json',**workerpin),
        original_outcome=outpin,source_manifest=mfpin,original_intent=intentpin,
        current_version_evidence=version,original_step_classifications=intent['original_step_observations'],
        original_worker_identity={key:worker[key] for key in ('node','host','PID','startticks','boot_id','uid')},
        known_srun_reaped=True,read_only_independent_sacct_queries=1,independent_body_rehash=False,
        GPU_GO=False,still_requires_original_before_inputs=True)
    print(json.dumps(authority,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
