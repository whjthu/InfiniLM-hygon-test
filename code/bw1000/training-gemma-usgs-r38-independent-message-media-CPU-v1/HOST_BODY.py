
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
    raw = sys.stdin.buffer.read(190001)
    need(0 < len(raw) <= 190000, "BOUNDED_SOURCE_ONLY_PACKET")
    packet = json.loads(raw)
    need(set(packet) == {'claim','r27','sources'}, "EXACT_SOURCE_ONLY_PACKET")
    claim_raw = b64decode(packet['claim'],validate=True)
    previous_host = b64decode(packet['r27'],validate=True)
    sources = {key:b64decode(value,validate=True) for key,value in packet['sources'].items()}
    need(digest(claim_raw)==CLAIM_SHA and json.loads(claim_raw)==CLAIM and digest(previous_host)==R27_SHA,
         "EXACT_PRIOR_HOST_OR_CURRENT_CLAIM_SOURCE")
    need(set(sources)=={'probe.py', 'probe_r38.py', 'r30_cache_gate.py', 'r31_modelscope_cache.py', 'MODEL_PROCESSOR_FILES.json', 'ipc_witness.py', 'arrow_lineage.py', 'cleanup.py', 'SOURCE_FILES.json', 'prepare.py'}
         and {key:digest(raw) for key,raw in sources.items()}==CLAIM['sources'], "EXACT_RELEASED_SOURCE_BYTES")
    manifest = json.loads(sources['SOURCE_FILES.json'])
    need(manifest['files']=={n:{'bytes':len(sources[n]),'SHA256':digest(sources[n])} for n in ('probe.py', 'probe_r38.py', 'r30_cache_gate.py', 'r31_modelscope_cache.py', 'MODEL_PROCESSOR_FILES.json', 'ipc_witness.py', 'arrow_lineage.py', 'cleanup.py')},
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
    if 'native_first_two_rank_slots' in result:
        projected=result['native_first_two_rank_slots']
        need(type(projected) is dict and set(projected)=={'rank_projections','boundaries','slots','distinct_original_ordinals','private_sha256'}
             and tuple(projected[k] for k in ('rank_projections','boundaries','slots','distinct_original_ordinals'))==(8,2,64,64)
             and re.fullmatch(r'[a-f0-9]{64}',projected['private_sha256']),"EXACT_CPU_FIRST_TWO_BOUNDARY_COUNTS")
        public['native_first_two_rank_slots']=projected
    return public


IPC_PHASES=("swift_unshuffled_call","swift_shuffled_call")


def project_ipc(value):
    public={'policy':'REQUEST_PRIVATE_UNIX_EXACT_PHASE_ENDPOINT','raw_addresses_forwarded':False}
    if value=={**public,'summary':'UNAVAILABLE'}:
        return value
    suffix={'phase_metrics','phases_closed','state_failure','private_witness_SHA256','private_witness_bytes','owner_diagnostic_failures'}
    need(type(value) is dict and set(value)==set(public)|suffix and all(value[k]==v for k,v in public.items())
         and type(value['phases_closed']) is int and 0<=value['phases_closed']<=2
         and type(value['private_witness_bytes']) is int and 1<=value['private_witness_bytes']<=32768
         and type(value['owner_diagnostic_failures']) is int and 0<=value['owner_diagnostic_failures']<=4294967295
         and re.fullmatch(r'[0-9a-f]{64}',value['private_witness_SHA256'])
         and value['state_failure'] in (None,'AUTHORIZED_NATIVE_CONNECT_FAILED','PRIVATE_BIND_CANDIDATE_LIMIT',
                                        'PRIVATE_LISTEN_REPLACEMENT_OR_CROSS_PHASE_REUSE'),"BOUNDED_R34_PUBLIC_PHASE_STATE")
    phases=value['phase_metrics']
    need(type(phases) is dict and set(phases)==set(IPC_PHASES),"EXACT_R34_TWO_LOADER_PHASES")
    counts={'registered_endpoints':1,'initiator_identities':8,'permits':4294967295,'native_failed':4294967295,
            'native_returned':4294967295,'pending':8,'denied':4294967295}
    rejected=(None,'NOT_WRAPPED_NATIVE_CONNECT','OUTSIDE_SWIFT_LOADER','UNPROVEN_MULTIPROCESS_CALLSITE',
              'NOT_PRIVATE_UNIX_STREAM_PATHNAME','SHARED_LOADER_PHASE_UNPROVEN','LIVE_INITIATOR_STEP_UNPROVEN',
              'PRIVATE_REGISTERED_LISTENER_UNPROVEN','PRIVATE_PATHNAME_IDENTITY_CHANGED','PROCESS_SCOPE_OR_PENDING_CALL_LIMIT',
              'EVIDENCE_OR_DEADLINE_UNAVAILABLE')
    for phase in phases.values():
        need(type(phase) is dict and set(phase)==set(counts)|{'chain_SHA256','last_rejection'}
             and all(type(phase[k]) is int and 0<=phase[k]<=maximum for k,maximum in counts.items())
             and phase['last_rejection'] in rejected and re.fullmatch(r'[0-9a-f]{64}',phase['chain_SHA256'])
             and phase['permits']==phase['native_returned']+phase['native_failed']+phase['pending'],
             "EXACT_BOUNDED_R34_PHASE_METRICS")
    return value


def verify_ipc(read,run,value,step,scope):
    value=project_ipc(value)
    need(value['phases_closed']==2 and value['state_failure'] is None and value['owner_diagnostic_failures']==0,
         "EXACT_R34_CLOSED_CLEAN_IPC_PHASES")
    raw=read(run/'out/PRIVATE_IPC_STATE.json',value['private_witness_bytes'],value['private_witness_SHA256'],cap=32768)
    saved=json.loads(raw)
    need((saved['request'],saved['step'],saved['schema'],saved['active'],saved['next_phase'],saved['failure'])==
         (scope,step,'R34_BOUNDED_PRIVATE_IPC_STATE_V1',None,2,None) and saved['root']['same_slurm_step'] is True
         and set(saved['phases'])==set(IPC_PHASES),"EXACT_PRIVATE_R34_IPC_STATE_SCOPE")
    seen=[]
    for name in IPC_PHASES:
        row=saved['phases'][name]
        public=value['phase_metrics'][name]
        holder=row['endpoint']
        need(row['issued']==row['returned']==public['permits']==public['native_returned']
             and row['failed']==public['native_failed']==public['pending']==0
             and all(token==0 for token in row['initiators'].values()) and len(row['initiators'])==public['initiator_identities']
             and len(row['candidates'])<=8 and row['chain_SHA256']==public['chain_SHA256'] and row['denied']==public['denied']
             and row['rejection']==public['last_rejection'] and int(holder is not None)==public['registered_endpoints'],
             "EXACT_R34_PER_PHASE_RETURNS_AND_PRIVATE_COUNTS")
        if holder is not None:
            file=holder['file']['identity']
            actor=holder['process']
            root=saved['root']
            identity=(file['device'],file['inode'],file['uid'])
            need(holder in row['candidates'] and holder['file']['category']=='PRIVATE_REQUEST_SOCKET'
                 and actor['same_slurm_step'] is True and actor['uid']==root['uid']==file['uid']
                 and actor['namespace_inode']==root['namespace_inode']
                 and all(path_hash!=holder['address_SHA256'] and former!=identity for path_hash,former in seen),
                 "EXACT_SEPARATE_PRIVATE_LOADER_ENDPOINTS")
            seen.append((holder['address_SHA256'],identity))
        else:
            need(row['issued']==0,"NO_NATIVE_PERMIT_WITHOUT_PHASE_ENDPOINT")
    total=sum(row['native_returned'] for row in value['phase_metrics'].values())
    need(total>0,"AT_LEAST_ONE_ACTUAL_IPC_RETURN_REQUIRED")
    return raw,total


R36_NATIVE_PROVIDER_PINS={'swift_preprocessor': {'bytes': 24438, 'SHA256': '55f4751391583048bb9c4eca90f95b5d95923c5d958edd0ffe37be223438fa33'}, 'datasets_json': {'bytes': 98124, 'SHA256': 'd52713f02c9e8f94b085a0e7ca29dc06d66d110e2500806a5541a891519049bc'}, 'datasets_formatter': {'bytes': 26626, 'SHA256': 'f3332fedd7d3cb2103bf3363283cd3ec46f6fdf9f69fa3a49df1a660aff4f8c3'}}


def project_independent_passes(evidence):
    states={"NOT_REACHED","SETUP_BLOCKED","INCOMPLETE_BUDGET","VALUE_OR_DECODER_MISMATCH","COMPLETE"}
    output={}
    for name in ("message_pass","media_pass"):
        value=evidence[name]
        assigned,seen,qualified=(value[k] for k in ("planned_rows","rows_examined","qualified_rows"))
        state=value["state"]
        diff=value["first_difference"]
        need(type(assigned) is type(seen) is type(qualified) is int and assigned==800
             and 0<=qualified<=seen<=800 and state in states
             and (state!="COMPLETE" or seen==qualified==800 and diff is None),"R38_EXACT_DUAL_PASS_SCOPE")
        if name=="message_pass":
            need(value["planned_elements"]==1600 and value["qualified_elements"]==2*qualified,"R38_EXACT_MESSAGE_ELEMENTS")
        elif state=="COMPLETE":
            declared=value["image_feature_declarations"]
            need(set(declared)=={"swift_unshuffled_call","swift_shuffled_call"}
                 and all(item["physical_type_matches"] is True and re.fullmatch(r"[0-9a-f]{64}",item["nested_feature"]["type_SHA256"])
                         for item in declared.values()),"R38_DECLARED_AND_PHYSICAL_MEDIA_SCHEMA_EVIDENCE")
        projected={"state":state,"planned_rows":assigned,"rows_examined":seen,"qualified_rows":qualified,"first_difference_ordinals":None}
        if diff is not None:
            keys=("physical_ordinal","unshuffled_logical_ordinal","shuffled_logical_ordinal")
            need(all(type(diff[k]) is int and 1<=diff[k]<=800 for k in keys)
                 and diff["physical_ordinal"]==diff["unshuffled_logical_ordinal"],"R38_DUAL_PASS_FIRST_DIFFERENCE_ORDINALS")
            projected["first_difference_ordinals"]={k:diff[k] for k in keys}
        output[name]=projected
    return output


def read_audits(read,run,native,row,scope):
    result={"planned_arrow_rows":800,"qualified_arrow_rows":0,"planned_message_elements":1600,"qualified_message_elements":0,"planned_media_rows":800,"qualified_media_rows":0,"planned_ranks":8,"planned_first_two_slots":64,
            "arrow":"MISSING_OR_INVALID","cleanup":"MISSING_OR_INVALID","historical_3369_cleanup":"ENOTEMPTY"}
    def receipt(key,name,schema):
        pin=native[key]
        need(set(pin)=={"bytes","SHA256"} and type(pin["bytes"]) is int and re.fullmatch(r"[0-9a-f]{64}",pin["SHA256"]),
             "BOUNDED_R35_EVIDENCE_IDENTITY")
        value=json.loads(read(run/'out'/name,pin["bytes"],pin["SHA256"],cap=16384))
        need((value["step"],value["request"],value["schema"])==(row[0],scope,schema),"R35_SAME_TERMINAL_AUDIT_SCOPE")
        return pin,value
    try:
        pin,evidence=receipt("arrow_audit","ARROW_AUDIT.json","R35_ARROW_LINEAGE_NO_ROW_VALUES_V1")
        count=evidence["qualified_rows"]
        messages=evidence["qualified_message_elements"]
        images=evidence["qualified_media_rows"]
        independent=project_independent_passes(evidence)
        mr=independent["message_pass"]["qualified_rows"]
        ir=independent["media_pass"]["qualified_rows"]
        need(type(count) is type(messages) is type(images) is int and messages==2*mr and images==ir
             and max(0,mr+ir-800)<=count<=min(mr,ir) and evidence["planned_message_elements"]==1600
             and evidence["planned_media_rows"]==800,"R38_INDEPENDENT_FIXED_DENOMINATORS")
        need(evidence["planned_rows"]==800 and type(count) is int and 0<=count<=800
             and evidence["prior_3369_cleanup"]=="ENOTEMPTY","R35_ARROW_FIXED_DENOMINATOR")
        if evidence.get("first_difference") is not None:
            difference=evidence["first_difference"]
            ordinals=("physical_ordinal","unshuffled_logical_ordinal","shuffled_logical_ordinal")
            need(all(type(difference[k]) is int and 1<=difference[k]<=800 for k in ordinals),"EXACT_FIRST_DIFFERENCE_ORDINALS")
            result["first_difference_ordinals"]={key:difference[key] for key in ordinals}
        clone=evidence.get("native_formatter_clone",{})
        clone_flags={"only_messages","native_decoder_shares_features","messages_decoding_enabled","outer_list_rebuilt",
                     "native_method_implementations_matched","storage_and_scoped_arrow_types_matched",
                     "original_json_leaf_retained_by_native_visitor"}
        clone_ok=(type(clone) is dict and set(clone)==clone_flags|{"native_config_fields_SHA256"}
                  and all(clone[k] is True for k in clone_flags)
                  and re.fullmatch(r"[0-9a-f]{64}",clone["native_config_fields_SHA256"]) is not None)
        result["verified_native_formatter_clone"]=clone_ok
        metadata=evidence["loader_metadata"]
        need(set(metadata)<={"swift_unshuffled_call","swift_shuffled_call"},"R35_EXACT_LOADER_METADATA_KEYS")
        result.update(qualified_arrow_rows=count,qualified_message_elements=messages,qualified_media_rows=images,
                      **independent,arrow_SHA256=pin["SHA256"],metadata_snapshots=len(metadata),
                      arrow="COMPLETE" if count==images==800 and messages==1600 and clone_ok and evidence["runtime_pins_verified"] is True
                      and all(val["state"]=="COMPLETE" for val in independent.values())
                      and evidence["actual_json_runtime_providers"]==R36_NATIVE_PROVIDER_PINS
                      and evidence["matched_complete"] is True and evidence["permutation_verified"] is True
                      and len(metadata)==2 and all(x["state"]=="AVAILABLE" for x in metadata.values()) else "INCOMPLETE")
    except (OSError,ValueError,KeyError,TypeError,AttributeError):
        pass
    try:
        pin,evidence=receipt("cleanup_audit","CLEANUP_AUDIT.json","R35_REQUEST_PRIVATE_TEARDOWN_V1")
        states={"COMPLETE","SCRATCH_NOT_CREATED","EXACT_PRIVATE_SCRATCH_IDENTITY_UNPROVEN","OWNED_NATIVE_CHILDREN_NOT_RELEASED",
                "SCRATCH_MMAP_OR_FD_RELEASE_UNPROVEN","REQUEST_SCRATCH_REMAINS","TEARDOWN_EVIDENCE_INCOMPLETE"}
        need(evidence["state"] in states and evidence["prior_3369_ENOTEMPTY_retained"] is True,"R35_SEPARATE_CLEANUP_STATE")
        attempts=evidence.get("attempts",[])
        need(type(attempts) is list and len(attempts)<=2 and all(type(x) is str and re.fullmatch(r'[A-Z0-9_]{1,40}',x) for x in attempts),
             "BOUNDED_R35_CLEANUP_ERRORS")
        result.update(cleanup=evidence["state"] if evidence["state"]!="COMPLETE" or evidence["removed"] is True else "MISSING_OR_INVALID",
                      cleanup_SHA256=pin["SHA256"],cleanup_attempt_classes=attempts)
    except (OSError,ValueError,KeyError,TypeError,AttributeError):
        pass
    return result


def good_audits(value):
    return value["arrow"]=="COMPLETE" and value["cleanup"]=="COMPLETE"


def verify_private_projection(read,run,native,row,scope):
    need(row[2:4]==['COMPLETED','0:0'] and int(row[4])<=240 and native['passed'] is True
         and native['native_loader_completed']=={'swift_unshuffled_call':True,'swift_shuffled_call':True}
         and native['arrow_reached'] is native['sampler_reached'] is native['scratch']['removed'] is True
         and native['actual_swift_physical_rows']==native['actual_physical_slots']==800 and native['actual_rank_projections']==8,
         "NATIVE_CPU_C0_ARROW_AND_RANK_PROJECTION_REQUIRED")
    evidence,native_connects=verify_ipc(read,run,native['ipc_boundary'],row[0],scope)
    meta=native['private_projection']
    raw=read(run/'out/PRIVATE_ORDER.json',meta['bytes'],meta['SHA256'],cap=65536)
    order=json.loads(raw)
    need((order['request'],order['step'],order['baseline_view_SHA256'],order['production_adoption'])==
         (scope,row[0],'74fd9aad043441b7e29578223605a0030a93c5c9614057707b381175cda9a0cc',False),
         "EXACT_PRIVATE_NATIVE_ORDER_IDENTITY")
    lineage=order['raw_ordinals_by_dataset_index']
    need(len(lineage)==800 and all(type(value) is int for value in lineage) and sorted(lineage)==list(range(1,801))
         and lineage!=list(range(1,801)),"ALL_800_ACTUAL_ARROW_PHYSICAL_ORDINALS")
    ranks=order['ranks']
    need(set(ranks)=={str(rank) for rank in range(8)} and all(type(v) is list and len(v)==100 for v in ranks.values())
         and sorted(slot for r in ranks.values() for slot in r)==list(range(1,801)),"ALL_EIGHT_NATIVE_RANK_PROJECTIONS")
    first={r:[v[:4],v[4:8]] for r,v in ranks.items()}
    first_hash=digest((json.dumps(first,sort_keys=True,separators=(',',':'))+'\n').encode())
    need(order['two_optimizer_boundaries_projection_only']==first and native['native_first_two_rank_slots']['private_sha256']==first_hash,
         "EXACT_FIRST_TWO_NATIVE_ACCUMULATION_BOUNDARIES")
    return {'accepted':True,'arrow_physical_rows':800,'rank_projections':8,'first_two_slots':64,
            'native_connects':native_connects,'qualified_message_elements':1600,'qualified_media_rows':800,'private_projection_SHA256':digest(raw),'private_ipc_SHA256':digest(evidence),
            'actual_distributed_ranks':0,'optimizer_updates':0}


def main():
    scope,parent,node,label = (CLAIM[key] for key in ('request','parent','node','label'))
    need(os.getuid()==154459 and (parent,node)==('1103606','f13r1n12')
         and label=='training-gemma-usgs-r38-independent-message-media-CPU-v1'
         and CLAIM['target_GPU']==0 and CLAIM['CPU']==4 and CLAIM['memory_MiB']==8192
         and CLAIM['work_seconds']==240 and CLAIM['max_target_steps']==1 and CLAIM['replay_allowed'] is False,
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
                result['state']='INDEPENDENT_R38_TERMINAL_SCOPE_OBSERVED'
                result['cpu_admission']={'accepted':False,'diagnostic':'NATIVE_CPU_PROJECTION_INCOMPLETE'}
                if native:
                    result['qualification_IO']=read_audits(read,run,native,row,scope)
                if native and native.get('passed') is True and good_audits(result['qualification_IO']):
                    try:
                        result['cpu_admission']=verify_private_projection(read,run,native,row,scope)
                    except (OSError,ValueError,KeyError,TypeError,AttributeError):
                        result['cpu_admission']={'accepted':False,'diagnostic':'PRIVATE_NATIVE_ACCEPTANCE_EVIDENCE_REJECTED'}
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
    for name in ('probe.py', 'probe_r38.py', 'r30_cache_gate.py', 'r31_modelscope_cache.py', 'MODEL_PROCESSOR_FILES.json', 'ipc_witness.py', 'arrow_lineage.py', 'cleanup.py', 'SOURCE_FILES.json'):
        with (run/'src'/name).open('xb') as file:
            file.write(sources[name])
            file.flush()
            os.fsync(file.fileno())
    now=time.time()
    command=argv['argv'](parent,node,scope,now,CLAIM['sources']['SOURCE_FILES.json'],CLAIM['allow_component_fallback'])
    need(command[0]=='srun' and '--gres=none' in command and '--overlap' not in command
         and '--cpus-per-task=4' in command and '--mem=8192M' in command and now+240<=placement['parent_end_epoch'],
         "EXACT_ZERO_GRES_ARGV_AND_EPOCH")
    binds=[Path(command[i+1].split(':',1)[0]) for i,arg in enumerate(command) if arg=='--bind']
    need(len(binds)==13 and all(p.resolve(strict=True)==p and (p.is_dir() or p.is_file()) for p in binds)
         and Path(command[-6]).is_dir() and Path(command[command.index('--time=00:04:00')+1]).is_file(),
         "EXACT_REAL_SAMPLER_MOUNTS")
    intent={'scope_request':scope,'publication':PUBLICATION,'epoch':now,'deadline':now+240,'parent':parent,'node':node,
            'argv_SHA256':digest(host['doc'](command)), 'claim_SHA256':CLAIM_SHA,'placement':placement,'replay_allowed':False}
    host['save'](run,'HOST_INTENT.json',intent)
    attempt=False
    try:
        with (run/'HOST.stdout').open('xb') as out,(run/'HOST.stderr').open('xb') as err:
            remaining=now+240-time.time()
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
