
from base64 import b64decode
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

def need(condition, reason):
    if not condition:
        raise ValueError(reason)

def digest(raw):
    return sha256(raw).hexdigest()

def original_packet():
    raw = sys.stdin.buffer.read(130001)
    need(0 < len(raw) <= 130000, "BOUNDED_SOURCE_ONLY_PACKET")
    packet = json.loads(raw)
    need(set(packet) == {'claim','r27','sources'}, "EXACT_SOURCE_ONLY_PACKET")
    claim_raw = b64decode(packet['claim'],validate=True)
    previous_host = b64decode(packet['r27'],validate=True)
    sources = {key:b64decode(value,validate=True) for key,value in packet['sources'].items()}
    need(digest(claim_raw)==CLAIM_SHA and json.loads(claim_raw)==CLAIM and digest(previous_host)==R27_SHA,
         "EXACT_PRIOR_HOST_OR_CURRENT_CLAIM_SOURCE")
    need(set(sources)=={'probe.py', 'probe_r32.py', 'r30_cache_gate.py', 'r31_modelscope_cache.py', 'MODEL_PROCESSOR_FILES.json', 'ipc_witness.py', 'SOURCE_FILES.json', 'prepare.py'}
         and {key:digest(raw) for key,raw in sources.items()}==CLAIM['sources'], "EXACT_RELEASED_SOURCE_BYTES")
    manifest = json.loads(sources['SOURCE_FILES.json'])
    need(manifest['files']=={n:{'bytes':len(sources[n]),'SHA256':digest(sources[n])} for n in ('probe.py', 'probe_r32.py', 'r30_cache_gate.py', 'r31_modelscope_cache.py', 'MODEL_PROCESSOR_FILES.json', 'ipc_witness.py')},
         "EXACT_COMPUTE_MANIFEST_MEMBERS")
    host = {'__name__':'r27_admission_reused'}
    exec(compile(previous_host,'<R27 published HOST functions>','exec'),host)
    argv = {'__name__':'r28_pinned_command', '__file__':'/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/' + CLAIM['label'] + '/' + CLAIM['request'] + '/prepare.py'}
    exec(compile(sources['prepare.py'],'<R28 published CPU command>','exec'),argv)
    return host,argv,sources

def select_account(query, argv, job, parent, node):
    all_rows=[line.split('|') for line in query('sacct','-nP','-j',parent,'--format',argv['ACCOUNTING'])]
    need(all(len(row)==9 for row in all_rows),"EXACT_ACCOUNTING_COLUMNS")
    rows=[row for row in all_rows if row[1]==job]
    need(len(rows)<=1 and all(re.fullmatch(parent+r'[.][0-9]+',row[0]) and row[5]==node for row in rows),
         "EXACT_ORIGINAL_ACCOUNTING_IDENTITY")
    return rows

def project(result):
    whitelist=('route','current_phase','phases','passed','step','actual_swift_physical_rows','actual_gpu',
               'actual_optimizer_updates','actual_training_ranks','training_ready','production_adoption',
               'diagnostic','original_sft_binding_diagnosis','scratch','input_environment','builder_cache_evidence','sft_cpu_validation_category','original_sft_argument_identity','modelscope_cache','native_loader_completed','arrow_reached','sampler_reached')
    public={key:result[key] for key in whitelist if key in result}
    if 'ipc_boundary' in result:
        public['ipc_boundary']=project_ipc(result['ipc_boundary'])
    return public


def project_ipc(value):
    keys=('denied_connects_observed','binds_observed','listens_observed','diagnostic_failures_in_owner','private_witness_bytes')
    result={'policy':'OBSERVE_AND_DENY_ALL_ORIGINAL_CONNECTS','raw_addresses_forwarded':False}
    if value==(result | {'summary':'UNAVAILABLE'}):
        return value
    need(type(value) is dict and set(value)==set(keys)|set(result)|{'last_observed_attempt','private_witness_SHA256'}
         and all(value[k]==v for k,v in result.items())
         and all(type(value[k]) is int and 0<=value[k]<=32768 for k in keys),"BOUNDED_R32_IPC_SUMMARY")
    result.update({k:value[k] for k in keys})
    sha_pattern=r'[a-f0-9]{64}'
    need(type(value['private_witness_SHA256']) is str and re.fullmatch(sha_pattern,value['private_witness_SHA256']),
         "EXACT_R32_PRIVATE_WITNESS_HASH")
    result['private_witness_SHA256']=value['private_witness_SHA256']
    detail=value['last_observed_attempt']
    if detail is not None:
        patterns={'family':r'AF_[A-Z0-9_]{1,40}|UNRECOGNIZED','socket_type':r'SOCK_[A-Z0-9_]{1,40}|UNRECOGNIZED',
                  'address_kind':r'UNIX_(?:ABSTRACT_BYTES|ABSTRACT_TEXT|PATHNAME)|INET_(?:LOOPBACK|UNSPECIFIED|OTHER|HOSTNAME_OR_NONLITERAL)|BYTES|TEXT|MISSING|UNSUPPORTED_SHAPE',
                  'caller_category':r'MULTIPROCESSING_CONNECTION|MULTIPROCESSING_MANAGER|PYTHON_SOCKET|DATASETS|SWIFT|MODELSCOPE|MULTIPROCESS|OTHER',
                  'initiator':r'SAME_REQUEST_SLURM_STEP_VERIFIED|UNCONFIRMED',
                  'peer':r'CORRELATED_LIVE_SAME_REQUEST_LISTENER|UNCONFIRMED','callsite_SHA256':sha_pattern,
                  'filesystem_endpoint':r'NOT_PATHNAME|UNKNOWN|UNSAFE_PATH_COMPONENT|OUTSIDE_REQUEST_SCRATCH|NONPRIVATE_REQUEST_PARENT|NONSOCKET_REQUEST_ENDPOINT|OTHER_UID_REQUEST_ENDPOINT|PRIVATE_REQUEST_SOCKET',
                  'same_as_successful_bind_endpoint':r'MATCH|DIFFERENT|UNKNOWN',
                  'same_as_successful_listen_endpoint':r'MATCH|DIFFERENT|UNKNOWN'}
        optional_shas={'address_SHA256','initiator_identity_SHA256','peer_identity_SHA256','filesystem_endpoint_SHA256'}
        need(type(detail) is dict and set(detail)==set(patterns)|optional_shas|{'connection_opened'}
             and detail['connection_opened'] is False
             and all(type(detail[k]) is str and re.fullmatch(pattern,detail[k]) for k,pattern in patterns.items())
             and all(detail[k] is None or (type(detail[k]) is str and re.fullmatch(sha_pattern,detail[k])) for k in optional_shas),
             "ALLOWLISTED_R32_IPC_DETAIL")
    result['last_observed_attempt']=detail
    return result


def main():
    scope,parent,node,label = (CLAIM[key] for key in ('request','parent','node','label'))
    need(os.getuid()==154459 and (parent,node)==('1103606','f13r1n12')
         and label=='training-gemma-usgs-r32-ipc-boundary-CPU-v1'
         and CLAIM['target_GPU']==0 and CLAIM['CPU']==4 and CLAIM['memory_MiB']==8192
         and CLAIM['work_seconds']==120 and CLAIM['max_target_steps']==1 and CLAIM['replay_allowed'] is False,
         "EXACT_PUBLISHED_R28_ZERO_GRES")
    host,argv,sources=original_packet()
    results=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
    run=results/label/scope
    job=label+'-'+scope[:12]
    host.update(JOB=job,REQUEST=scope,LABEL=label,RUN=run,STANDBY_STEP='1103606.3355',STANDBY_LABEL='training80-dcu-standby-4d63273bbce5-f13r1n12')
    env=host['host_environment']()
    os.umask(0o077)
    start=time.monotonic()
    def query(*args):
        need(args[0] in ('sacct','scontrol','squeue'),"READ_ONLY_SCHEDULER")
        limit=min(8,30-(time.monotonic()-start))
        need(limit>0,"BOUNDED_SCHEDULER_READ")
        call=subprocess.run(args,stdin=subprocess.DEVNULL,capture_output=True,check=True,timeout=limit,env=env)
        need(len(call.stdout)<=1048576,"BOUNDED_SCHEDULER_METADATA")
        return call.stdout.decode('ascii').splitlines()
    if MODE=='terminal':
        accounting=select_account(query,argv,job,parent,node)
        result={'state':'ORIGINAL_ACCOUNTING_NOT_VISIBLE','accounting':accounting,'replay_allowed':False}
        read=host['hash_read']
        native=None
        if os.path.lexists(run/'out/RESULT.json'):
            native_raw=read(run/'out/RESULT.json',cap=65536)
            native=json.loads(native_raw)
            need(native['request']==scope and native['node']==node,"SAME_NATIVE_REQUEST_IDENTITY")
            result.update(state='ORIGINAL_DIAGNOSTIC_RECORDED',worker=project(native),worker_SHA256=digest(native_raw))
        if accounting:
            row,=accounting
            if row[2].split()[0] in host['TERMINAL']:
                initial=json.loads(read(run/'HOST_INTENT.json'))
                need(initial['scope_request']==scope and initial['claim_SHA256']==CLAIM_SHA,"SAME_ORIGINAL_HOST_INTENT")
                result['independent']=argv['independent'](row,native,parent,node,scope,initial['epoch'],CLAIM['sources']['SOURCE_FILES.json'])
                result['state']='INDEPENDENT_R32_TERMINAL_SCOPE_OBSERVED'
        return result
    need(MODE=='launch' and not os.path.lexists(run) and not select_account(query,argv,job,parent,node),
         "ORIGINAL_ALREADY_PRESENT_RECONCILE_ONLY")
    placement=host['fresh_slots'](query,time.time())
    need(not select_account(query,argv,job,parent,node),"ORIGINAL_LABEL_VISIBLE_AFTER_PLACEMENT")
    run.parent.mkdir(exist_ok=True)
    need(run.parent.resolve(strict=True)==run.parent,"EXACT_CPU_RUN_PARENT")
    run.mkdir()
    (run/'src').mkdir()
    (run/'out').mkdir()
    (run/'scratch').mkdir()
    for name in ('probe.py', 'probe_r32.py', 'r30_cache_gate.py', 'r31_modelscope_cache.py', 'MODEL_PROCESSOR_FILES.json', 'ipc_witness.py', 'SOURCE_FILES.json'):
        with (run/'src'/name).open('xb') as file:
            file.write(sources[name])
            file.flush()
            os.fsync(file.fileno())
    now=time.time()
    command=argv['argv'](parent,node,scope,now,CLAIM['sources']['SOURCE_FILES.json'],CLAIM['allow_component_fallback'])
    need(command[0]=='srun' and '--gres=none' in command and '--overlap' not in command
         and '--cpus-per-task=4' in command and '--mem=8192M' in command and now+120<=placement['parent_end_epoch'],
         "EXACT_ZERO_GRES_ARGV_AND_EPOCH")
    binds=[Path(command[i+1].split(':',1)[0]) for i,arg in enumerate(command) if arg=='--bind']
    need(len(binds)==13 and all(p.resolve(strict=True)==p and (p.is_dir() or p.is_file()) for p in binds)
         and Path(command[-6]).is_dir() and Path(command[command.index('--time=00:03:00')+1]).is_file(),
         "EXACT_REAL_SAMPLER_MOUNTS")
    intent={'scope_request':scope,'publication':PUBLICATION,'epoch':now,'deadline':now+120,'parent':parent,'node':node,
            'argv_SHA256':digest(host['doc'](command)), 'claim_SHA256':CLAIM_SHA,'placement':placement,'replay_allowed':False}
    host['save'](run,'HOST_INTENT.json',intent)
    attempt=False
    try:
        with (run/'HOST.stdout').open('xb') as out,(run/'HOST.stderr').open('xb') as err:
            remaining=now+120-time.time()
            need(remaining>0,"ORIGINAL_CPU_EPOCH_EXPIRED")
            attempt=True
            child=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=remaining,env=env,check=False)
            observation=dict(state='CPU_SRUN_RETURNED_REQUIRES_INDEPENDENT_TERMINAL',child_rc=child.returncode)
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        observation=dict(state='CPU_SEND_OUTCOME_REQUIRES_RECONCILIATION',error_class=type(error).__name__)
    observation.update(attempted=attempt,placement=placement,replay_allowed=False)
    host['save'](run,'HOST_RETURN.json',observation)
    return observation

try:
    report=main()
except (OSError,ValueError,KeyError,TypeError,AttributeError,subprocess.SubprocessError) as fault:
    report={'state':'CPU_REQUEST_BLOCKED_RECONCILE_IF_INTENT_EXISTS','error_class':type(fault).__name__,
            'reason':str(fault) if type(fault) is ValueError and re.fullmatch(r'[A-Z0-9_]+',str(fault)) else None,
            'replay_allowed':False}
print(json.dumps({**report,'request':ID,'scope_request':CLAIM['request']},sort_keys=True,allow_nan=False),flush=True)
