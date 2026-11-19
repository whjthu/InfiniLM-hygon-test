"""Offline new dev50+blind50 terminal/300-slot audit. No labels or state tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import re

REVIEWED_CALLER_SHA256='b50f61c2a2659eb6788373cf6c7006fcb51b4f05f9cbbac96b7aed3fb4204803'
ARMS=('initial','domain25','generic25')
H=lambda b:hashlib.sha256(b).hexdigest()


def need(ok,why):
    if not ok:raise ValueError(why)


def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def pci(x):
    m=re.fullmatch(r'([0-9a-fA-F]{4}|[0-9a-fA-F]{8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])',x)
    need(m is not None,'PCI_FORMAT')
    d,b,s,f=(int(v,16) for v in m.groups());need(d<=65535 and s<=31,'PCI_RANGE')
    return d,b,s,f


def auxiliary_fd_scope(snapshot, supervisor):
    f=snapshot['own_UID_FD_snapshot']
    need(f['selected_device_or_KFD_holders']==[] and f['FD_observation_is_auxiliary'] is True
         and snapshot['primary_GPU_admission_satisfied'] is True and snapshot['scope_admission_satisfied'] is True
         and snapshot['scope_policy']=='primary_GPU_admission_satisfied_with_auxiliary_FD_coverage'
         and snapshot['FD_observation_is_auxiliary'] is True,'PRIMARY_GPU_WITH_AUXILIARY_FD')
    need(snapshot['exclusive_GPU_proven'] is False and f['exclusive_GPU_proven'] is False
         and f['all_FD_clean_qualified'] is False and snapshot['exe_identity']==f['exe_identity']=='UNKNOWN'
         and snapshot['transport_FDs_inspected'] is None and f['transport_FDs_inspected'] is None
         and snapshot['nontransport_UID_FD_complete'] is None and f['nontransport_UID_FD_complete'] is None,
         'NO_EXE_ANCESTOR_OR_EXCLUSIVE_CLAIM')
    total=f['incomplete_total'];records=f['incomplete_records'];truncated=f['incomplete_truncated_count']
    need(type(total)is int and total>=0 and total==snapshot['FD_incomplete_total']
         and isinstance(records,list) and len(records)<=64 and type(truncated)is int
         and truncated==total-len(records)>=0 and f['incomplete_truncated'] is (truncated>0),'FD_UNKNOWN_FULL_DENOMINATOR')
    size=len(json.dumps(records,ensure_ascii=True,sort_keys=True,allow_nan=False).encode())
    need(size<=16384 and size==f['incomplete_saved_bytes'],'FD_DIAGNOSTIC_BOUND')
    for k in ('completed_own_UID_processes','visited_processes','attempted_own_or_unknown_UID_processes'):
        need(type(f[k])is int and f[k]>=0,'FD_COUNTER_TYPES')
    need(f['process_attempt_limit']==4096 and f['global_visit_limit']==8192
         and f['checked_own_UID_processes']==f['attempted_own_or_unknown_UID_processes']<=4096
         and type(f['incomplete_processes'])is int and 0<=f['incomplete_processes']<=total
         and f['incomplete_processes']==snapshot['FD_incomplete_processes']
         and f['completed_own_UID_processes']+f['incomplete_processes']==f['attempted_own_or_unknown_UID_processes']
         and f['partial_process_attempts']==f['incomplete_processes']
         and f['attempted_own_or_unknown_UID_processes']<=f['visited_processes']<=8192,'FD_ATTEMPT_AND_VISIT_BOUNDS')
    need(all(type(f[k])is int and f[k]>=0 for k in ('permission_incomplete_total','gone_or_closed_race_total'))
         and f['permission_incomplete_total']+f['gone_or_closed_race_total']==total,'PERMISSION_AND_LIFECYCLE_DENOMINATORS')
    need(len({x['PID'] for x in records})<=f['incomplete_processes'],'UNIQUE_INCOMPLETE_PROCESS_COUNT')
    if not truncated:
        need(len({x['PID'] for x in records})==f['incomplete_processes']
             and sum(x['error_type']=='PermissionError' for x in records)==f['permission_incomplete_total'],'COMPLETE_DIAGNOSTIC_COUNTS')
    for x in records:
        need(type(x['PID'])is int and x['PID']>0 and x['error_type'] in ('PermissionError','FileNotFoundError','ProcessLookupError')
             and x['stage'] in ('proc_owner','proc_stat','fd_metadata','fd_directory','fd_target','after_proc_stat'),
             'ONLY_PERMISSION_OR_LIFECYCLE_COVERAGE_UNKNOWN')
        need(x['original_PID_termination_proven'] is False and isinstance(x['metadata_errors'],list) and len(x['metadata_errors'])<=2,'NO_LIFECYCLE_TERMINAL_INFERENCE')
        for e in x['metadata_errors']:
            need(e['stage'] in ('recover_after_proc_stat','recover_fd_metadata') and e['error_type'] in ('PermissionError','FileNotFoundError','ProcessLookupError'),'BOUNDED_RECOVERY_DIAGNOSTIC')
        for k in ('startticks','PPID','uid','errno','fd_uid','fd_mode','fd_number'):
            need(x[k] is None or type(x[k])is int and x[k]>=0,'PASSIVE_PARTIAL_IDENTITY_SCHEMA')
        before,after=x['identity_before'],x['identity_after']
        for z in (before,after):
            if z is not None:
                need(z['PID']==x['PID'] and type(z['startticks'])is int and z['startticks']>0
                     and type(z['PPID'])is int and z['PPID']>=0 and type(z['uid'])is int and z['uid']>=0,
                     'AVAILABLE_FD_PROCESS_IDENTITY')
        if before is not None and after is not None:
            need(before==after and x['identity_stable'] is True,'FD_IDENTITY_CHANGE_NOT_AUXILIARY')
        else:need(x['identity_stable'] is None,'UNAVAILABLE_IDENTITY_REMAINS_UNKNOWN')
    partial=total>0
    state='PARTIAL_UNKNOWN' if partial else 'COMPLETE_OWN_UID_READABLE'
    need(f['FD_coverage_state']==snapshot['FD_coverage_state']==state
         and f['all_UID_FD_complete'] is (not partial) and snapshot['all_UID_FD_complete'] is (not partial),
         'ACTUAL_FD_COVERAGE_STATE')
    return dict(FD_coverage_state=state,incomplete_total=total,incomplete_processes=f['incomplete_processes'],incomplete_saved=len(records),incomplete_truncated=truncated,
                permission_incomplete_total=f['permission_incomplete_total'],gone_or_closed_race_total=f['gone_or_closed_race_total'],
                completed_own_UID_processes=f['completed_own_UID_processes'],attempted_own_or_unknown_UID_processes=f['attempted_own_or_unknown_UID_processes'],
                all_UID_FD_complete=not partial,all_FD_clean_qualified=False,exclusive_GPU_proven=False,
                exe_identity='UNKNOWN',FD_observation_is_auxiliary=True,
                diagnostic_scope='Saved bounded records and producer total/truncated counters; unsaved entries not independently inspected')


def audit(bundle,context):
    files=bundle['files'];errors=[];missing=[]
    result=dict(context=context,original_execution_state=bundle['original_execution_state'],
        planned_generations=300,arms={a:dict(denominator=100,completed=0,missing=100,verified=0) for a in ARMS},
        rows=[dict(arm=a,ordinal=i,state='UNOBSERVED',natural_EOS=None,cap_hit=None) for a in ARMS for i in range(100)],
        engineering_metadata_verified=False,independent_EOS_metadata_verified=False,
        gold_read=False,scored=False,accuracy=None,state_body_reads=0,Torch_imports=0,device_queries=0,
        original_T04_domain_training_failure_retained=True,original_30_or_Model_replay_allowed=False,
        original_initial_or_checkpoint_independent_body_audit=False,formal17=False,effect_qualified=False)
    def get(name,required=True):
        x=files.get(name)
        if not x or 'value' not in x:
            if required:missing.append(name)
            return None
        return x['value']
    def check(label,fn):
        try:return fn()
        except (ValueError,KeyError,TypeError,IndexError,AttributeError) as e:
            errors.append(dict(boundary=label,error_type=type(e).__name__,constant=str(e)[:100] if type(e)is ValueError else None))
            return None
    def gate():
        need(bundle['context']==context and all(bundle[k]==0 for k in ('device_queries','Torch_imports','state_body_reads','gold_reads','remote_writes')),
             'READONLY_CONTEXT')
        need(not bundle['errors'] and bundle['original_execution_state'] in ('TERMINAL_PASS','TERMINAL_FAIL'), 'ORIGINAL_TERMINAL_UNOBSERVED')
        observations=bundle['original_process_observation']
        need(observations['processes'] and all(x['state'] in ('ABSENT','ORIGINAL_IDENTITY_ABSENT_PID_REUSED','EXITED_NOT_EXECUTING')
             for x in observations['processes']), 'ORIGINAL_OWNER_EXIT_NOT_PROVEN')
        return True
    original_outcome=get('run/outcome.json',False)
    if original_outcome:
        result['original_reported_failure']={k:original_outcome.get(k) for k in
            ('native_returncode','native_released','timed_out','engineering_completed','release_permitted')}
        failure=original_outcome.get('failure')
        if isinstance(failure,dict):
            result['original_reported_failure']['failure']={k:failure.get(k) for k in ('phase','error_type','message_SHA256')}
        result['failure_private_pin']={k:files.get('run/failure-private.json',{}).get(k) for k in ('bytes','SHA256')}
        result['process_identity_status']=bundle.get('original_process_identity','SEE_COLLECTOR_OBSERVATION')
    if not check('terminal_first',gate):
        result.update(errors=errors,missing_files=missing);return result
    if bundle.get('failed_before_native_release') is True:
        o=original_outcome;control=get('run/control-state.json');install=get('install-intent.json')
        def failure_gate():
            need(o['request_id']==control['request_id']==install['request_id']==context['request_id']
                 and o['publication']==control['publication']==install['publication']==context['publication']
                 and o['epoch']==control['epoch']==install['epoch']==context['epoch']
                 and o['source_manifest_SHA256']==control['source_manifest_SHA256']==install['source_manifest_SHA256']==context['source_manifest_SHA256']
                 and install['claim_SHA256']==context['claim_SHA256'],'FAILURE_CONTEXT')
            need(control['exec_request_attempted'] is False and control['entry_released'] is False
                 and o['native_released'] is False and o['engineering_completed'] is False
                 and o['native_returncode'] is None and o['planned_generations']==300
                 and o['deadline']==context['epoch']+5400,'FAILED_BEFORE_RELEASE')
            return True
        result.update(independent_failure_metadata_verified=bool(check('failed_admission',failure_gate)),
            post_status='UNKNOWN',predictions_observed=0,predictions_scored=0,errors=errors,missing_files=missing,
            prediction_read_gate='NOT_RELEASED_NO_PREDICTIONS',all_UID_FD_complete=False,
            exclusive_GPU_qualified=False,original_T04_domain_training_failure_retained=True)
        return result
    outcome=get('run/outcome.json');install=get('install-intent.json')
    mf=get('src/SOURCE_FILES.json');plan=get('src/core/PLAN.json');outer=get('src/INSTALL_PLAN.json')
    final=get('run/native/generation-finalized.json');generation=get('run/native/generation-plan.json')
    admission=get('run/worker-admission.json');owned=get('run/child-owner.json')

    def source():
        need(outcome['original_domain_training_accepted'] is False, 'ORIGINAL_T04_DOMAIN_TRAINING_FAILURE_RETAINED')
        need(context['namespace'] == '/home/whj/nvidia-baseline/alignment-assets/training-USGS-T04-wide-devblind300-effect-A100-v1', 'EXACT_CURRENT_OR_SAVED_FAILED_V2_SCOPE')
        need(outcome['request_id']==install['request_id']==context['request_id']
             and outcome['publication']==install['publication']==context['publication']
             and outcome['epoch']==install['epoch']==context['epoch']
             and outcome['deadline']==install['deadline']==context['epoch']+5400
             and outcome['finished_epoch']<=outcome['deadline'], 'ORIGINAL_CONTEXT')
        need(files['src/SOURCE_FILES.json']['SHA256']==context['source_manifest_SHA256']==outcome['source_manifest_SHA256'], 'ACTUAL_SOURCE_MF')
        for n in ('CLAIM.json','INSTALL_PLAN.json','core/SOURCE_FILES.json','core/PLAN.json','core/CLAIM.json'):
            need(files['src/'+n]['bytes']==mf['files'][n]['bytes'] and files['src/'+n]['SHA256']==mf['files'][n]['SHA256'], 'PUBLISHED_SOURCE_MEMBERS')
        need(mf['files']['core/infer_wide100.py']['SHA256']==REVIEWED_CALLER_SHA256,
             'EXACT_REVIEWED_INFERENCE_CALLER')
        need(outer['core_manifest_SHA256']==files['src/core/SOURCE_FILES.json']['SHA256']
             and mf['files']['qualified/bounded_GPU_supervisor.py']['SHA256']=='2307123f57756f3a7a42153176b2e3acb068f6e0a51e5756730468236bbf0e2f', 'CORE_RSS_BINDING')
        need(admission['request_id']==context['request_id'] and admission['epoch']==context['epoch']
             and admission['deadline']==context['epoch']+5250 and admission['whole_deadline']==context['epoch']+5400
             and admission['owned_child']==owned['child'] and admission['supervisor']==owned['supervisor']
             and admission['source_manifest_SHA256']==outer['core_manifest_SHA256'] and admission['CPU_affinity']==outer['CPU_affinity']==[6,7,8,9], 'ACTUAL_HELD_ADMISSION')
        before_record=get('run/before.json')
        need(admission['before']==before_record and admission['primary_GPU_admission_satisfied'] is True
             and admission['FD_observation_is_auxiliary'] is True and admission['all_FD_clean_qualified'] is False
             and admission['exclusive_GPU_proven'] is False
             and all(admission[k]==before_record[k] for k in ('scope_policy','all_UID_FD_complete','FD_coverage_state','FD_incomplete_total','FD_incomplete_processes')),'ACTUAL_ADMISSION_AUXILIARY_SCOPE')
        need(outcome['blind_inputs_and_media_read'] is True
             and outcome['input_media_observation']['inputs_and_media_read'] is True
             and outcome['input_media_observation']['evidence']['SHA256']==files['run/native/generation-plan.json']['SHA256'],
             'ACTUAL_INPUT_OBSERVATION_NOT_ADMISSION_ASSUMPTION')
        if final is None:
            need(files.get('run/native/generation-finalized.json', {}).get('exists') is False, 'FINALIZED_FILE_ABSENCE_MUST_BE_OBSERVED')
        for g in ((generation,) if final is None else (generation,final)):
            need(isinstance(g,dict), 'ACTUAL_GENERATION_METADATA_SCHEMA')
            need(g['arms']==list(ARMS) and g['per_arm']==100 and g['planned_generations']==300 and g['split_denominators']=={'dev':50,'blind':50} and g['generation_cap']==256
                 and g['input_SHA256']=={k:v['SHA256'] for k,v in plan['inputs'].items()}
                 and g['epoch']==context['epoch'] and g['deadline']==context['epoch']+5250
                 and g['dev_gold_read'] is False and g['blind_gold_read'] is False and g['blind_inputs_and_media_read'] is True
                 and g['domain_original_training_accepted'] is False and g['saved25_checkpoint_values_qualified'] is True and g['original_T04_domain_training_failure_retained'] is True, 'FIXED_THREE_ARM_INPUT_CONTRACT')
        pin=plan['native_identities'];ref=files['reference/native-identities']
        need(ref['bytes']==pin['bytes'] and ref['SHA256']==pin['SHA256'],'NATIVE_DEV_REFERENCE_PIN')
        need(len(ref['value'])==150,'ACTUAL_NATIVE150_REFERENCE')
        refs=[x for split in ('dev','blind') for x in ref['value'] if x['split']==split]
        need(len(refs)==100 and [x['ordinal'] for x in refs]==list(range(50))*2 and all(x['labels_None'] is True for x in refs), 'NEW_NATIVE100_REFERENCE')
        return refs
    refs=check('source_input_contract',source)

    def postgate():
        post=get('run/after.json');before=get('run/before.json');clean=outcome['cleanup']
        need(outcome['all_FD_clean_qualified'] is False and outcome['exclusive_GPU_proven'] is False
             and outcome['exe_identity']=='UNKNOWN' and outcome['scope_policy']==post['scope_policy']
             and outcome['transport_FDs_inspected'] is post['transport_FDs_inspected']
             and outcome['primary_GPU_admission_satisfied'] is True and outcome['FD_observation_is_auxiliary'] is True
             and outcome['scope_level_release_permitted'] is outcome['release_permitted']
             and all(outcome[k]==post[k] for k in ('FD_coverage_state','FD_incomplete_total','FD_incomplete_processes','all_UID_FD_complete')),'OUTCOME_HONEST_SCOPE')
        need(post==outcome['post'] and post['active_compute_processes']==0 and post['master_port_bindable'] is True
             and len(post['devices'])==len(before['devices'])==1, 'ACTUAL_SCOPE_POST')
        lease=get('run/cooperative-lease.json');lease_release=get('run/cooperative-lease-release.json');memory=get('run/host-memory-before.json')
        need(lease['request_id']==lease_release['request_id']==context['request_id'] and lease['owner']==owned['supervisor'] and lease['GPU_UUID']==outer['expected_GPU']['UUID'] and lease['LOCK_EX_NB'] is True and lease['cooperative_only'] is True and lease_release['owned_cleanup_and_post_qualified'] is True and lease_release['lock_file_deleted'] is False,'ACTUAL_COOPERATIVE_LEASE')
        need(memory['MemAvailable_bytes']>=32*1024**3 and memory['memory_reserved'] is False,'HOST_MEMORY_HEADROOM_ONLY')
        monitor=get('run/runtime-resource-observation.json');need(monitor['samples']>0 and monitor['minimum_HOST_MemAvailable_bytes']>=8*1024**3 and monitor['foreign_check_count']>0 and monitor['last_foreign_observation']['foreign_selected_GPU_PIDs']==[],'RUNTIME_HOST_MEMORY_FOREIGN_PID_SCOPE')
        result['FD_scope']={label:auxiliary_fd_scope(snapshot,owned['supervisor']) for label,snapshot in [('before',before),('after',post)]}
        result['post_status']='SCOPE_ADMISSION_SATISFIED_WITH_EXPLICIT_FD_COVERAGE'
        exp=outer['expected_GPU']
        for x in (before['devices'][0],post['devices'][0]):
            need(x['UUID']==exp['UUID'] and x['index']==exp['index'] and pci(x['PCI'])==pci(exp['PCI'])
                 and x['total_MiB']==40960 and x['free_MiB']>=38000,'ONE_ORIGINAL_A100')
        need(clean['owned_processes_remaining']==0 and clean['held_child']['reaped'] is True
             and outcome['release_permitted'] is True,'OWNED_RELEASE')
        return True
    clean=check('post',postgate)
    verified_total=0
    for arm in ARMS:
        values=get('run/native/'+arm+'-predictions-private.jsonl',False)
        values=[] if values is None else values
        def armgate():
            if final is None:
                need(not values, 'PREDICTIONS_WITHOUT_FINALIZED_SEAL_UNQUALIFIED')
                return False
            need(isinstance(final,dict), 'ACTUAL_FINALIZED_METADATA_SCHEMA')
            rec=final['arm_results'][arm]
            need(rec['planned']==100 and 0<=rec['completed']<=rec['attempted']<=100 and rec['missing']==100-rec['completed'], 'ARM_DENOMINATOR')
            need(isinstance(values,list) and len(values)==rec['completed'], 'SAVED_PREDICTION_COUNT')
            if 'file' in rec:
                need(rec['file']==arm+'-predictions-private.jsonl','EXACT_ARM_FILE')
                row=files['run/native/'+rec['file']]
                need(row['bytes']==rec['bytes'] and row['SHA256']==rec['SHA256'], 'FINALIZED_PREDICTION_PIN')
            need([x['ordinal'] for x in values]==list(range(len(values))), 'ORDERED_UNIQUE_SLOTS')
            return True
        ok=check(arm+'-file',armgate)
        count=len(values) if isinstance(values,list) and len(values)<=100 else 0
        result['arms'][arm].update(completed=count,missing=100-count,input_SHA256={k:v['SHA256'] for k,v in plan['inputs'].items()} if plan else None)
        for ordinal in range(100):
            row=result['rows'][ARMS.index(arm)*100+ordinal];row['state']='MISSING'
            returned=get('run/native/'+arm+'-native-return-%02d.json'%ordinal,False)
            row['native_return_present']=returned is not None
            if not ok or ordinal>=count:
                continue
            def slot():
                x=values[ordinal];need(refs is not None,'INPUT_REFERENCE_UNQUALIFIED')
                need(x['arm']==arm and x['status']=='COMPLETED' and x['ordinal']==ordinal
                     and H(x['id'].encode())==refs[ordinal]['source_id_SHA256']
                     and x['native_field_identity_verified'] is True and x['split']==refs[ordinal]['split'] and x['split_ordinal']==refs[ordinal]['ordinal'],'ACTUAL_INPUT_SLOT')
                need(returned is not None and returned['actual_native_infer_returned'] is True
                     and returned['qualified_prediction'] is False and returned['arm']==arm
                     and returned['ordinal']==ordinal and returned['id']==x['id'] and returned['split']==x['split'] and returned['split_ordinal']==x['split_ordinal']
                     and returned['raw_text']==x['text'] and returned['finish_reason']==x['finish_reason']
                     and returned['collator_calls']==returned['generate_calls']==1,'ORIGINAL_NATIVE_RETURN')
                ids=x['raw_generated_token_ids'];eos=x['raw_EOS_ids']
                need(isinstance(ids,list) and 0<len(ids)<=256 and all(type(t)is int and t>=0 for t in ids)
                     and isinstance(eos,list) and eos and all(type(t)is int and t>=0 for t in eos),'RAW_EOS_SCHEMA')
                need(ids==returned['raw_generated_token_ids'] and eos==returned['effective_EOS_ids'],'NATIVE_RAW_TOKEN_PROVENANCE')
                cap=len(ids)>=256;natural=ids[-1] in eos and not cap and not any(t in eos for t in ids[:-1])
                need(x['natural_EOS'] is natural and x['cap_hit'] is cap and x['generated_tokens']==len(ids)
                     and x['last_token_is_EOS'] is (ids[-1] in eos)
                     and x['generated_ids_SHA256']==H(canonical(ids)),'INDEPENDENT_EOS_CAP_RECOMPUTATION')
                row.update(state='VERIFIED',natural_EOS=natural,cap_hit=cap,generated_tokens=len(ids),
                           generated_ids_SHA256=H(canonical(ids)),text_SHA256=H(x['text'].encode()))
                return True
            if check(arm+'-%02d'%ordinal,slot):
                result['arms'][arm]['verified']+=1;verified_total+=1
    result.update(original_T04_domain_training_failure_retained=True,exclusive_GPU_qualified=False,
        errors=errors,missing_files=sorted(set(missing)),verified_predictions=verified_total,
        independent_EOS_metadata_verified=verified_total==300,
        source_and_input_plan_metadata_verified=refs is not None,
        generation_finalized_observed=isinstance(final,dict),
        prediction_seal_state='MISSING_FINALIZED_NATIVE_OUTPUT' if final is None else 'FINALIZED_METADATA_OBSERVED',
        independent_failed_without_predictions_metadata_verified=bool(refs is not None and clean and not errors
            and final is None and outcome['engineering_completed'] is False and outcome['native_returncode']==1
            and all(result['arms'][arm]['completed']==0 for arm in ARMS)),
        engineering_metadata_verified=bool(not errors and not missing and verified_total==300 and clean
            and outcome['engineering_completed'] is True and outcome['native_returncode']==0
            and not outcome['cleanup']['termination_needed'] and final['passed'] is True
            and final['all_predictions_finalized_before_gold'] is True
            and final['all_prediction_files_closed_before_gold'] is True),
        input_tensor_identity_scope='Saved slot IDs + published original native150/dev50/blind50 pin + frozen inference field-equality guard; no independent tensor re-encoding.',
        adapter_scope='Frozen caller requires actual selected504 torch.equal + same-base identity; no new adapter body read.',
        score_next='Only a separately authorized original score_after process may read gold after full300 prediction seal; this reader never scores.',
        original_small_files={n:{k:r[k] for k in ('bytes','SHA256','exists','error_type') if k in r} for n,r in files.items()})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('bundle','bundle-sha256','context','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();b=Path(a.bundle);need(b.stat().st_size<=32*1024**2,'BUNDLE32M')
    raw=b.read_bytes();need(H(raw)==a.bundle_sha256,'ORIGINAL_PRIVATE_BUNDLE_SHA')
    bundle=json.loads(raw);context=json.loads(Path(a.context).read_bytes())
    need(context['namespace']=='/home/whj/nvidia-baseline/alignment-assets/training-USGS-T04-wide-devblind300-effect-A100-v1','EXACT_CURRENT_OR_SAVED_FAILED_V2_SCOPE')
    result=audit(bundle,context)
    with Path(a.output).open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result.get(k) for k in ('original_execution_state','planned_generations','verified_predictions',
          'engineering_metadata_verified','gold_read','effect_qualified')}))
