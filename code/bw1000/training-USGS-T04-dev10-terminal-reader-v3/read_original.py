"""Offline original dev10 small-record audit. Never opens labels or state tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import re

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


def transport_scope(snapshot, supervisor):
    f=snapshot['own_UID_FD_snapshot']
    need(f['selected_device_or_KFD_holders']==[] and f['incomplete_processes']==0
         and f['nontransport_UID_FD_complete'] is True,'NONTRANSPORT_FD_ADMISSION')
    need(snapshot['scope_admission_satisfied'] is True and snapshot['exclusive_GPU_proven'] is False
         and snapshot['exe_identity']=='UNKNOWN' and snapshot['scope_policy']==f['scope_policy']
         and snapshot['all_UID_FD_complete'] is f['all_UID_FD_complete']
         and snapshot['transport_FDs_inspected'] is f['transport_FDs_inspected'],'SNAPSHOT_SCOPE_FLAGS')
    count=f['transport_exception_count'];need(type(count)is int and count in (0,1),'ONE_TRANSPORT_EXCEPTION_MAX')
    if count==0:
        need(f['all_UID_FD_complete'] is True and f.get('uninspected_transport_ancestor') is None,'FULLY_VISIBLE_FD_ROUTE')
        return dict(transport_exception_count=0,all_UID_FD_complete=True,exclusive_GPU_qualified=False)
    need(f['all_UID_FD_complete'] is False and f['transport_FDs_inspected'] is False
         and f['exe_identity']=='UNKNOWN' and f['scope_policy']=='scope_admission_satisfied_with_transport_exception',
         'EXPLICIT_UNINSPECTED_TRANSPORT_BOUNDARY')
    e=f['transport_evidence'];before,after=e['before'],e['after']
    need(e['stable_identity'] is True and e['status_identity_fields_stable'] is True
         and e['exe_identity']=='UNKNOWN' and e['transport_FDs_inspected'] is False,'TRANSPORT_EVIDENCE_FLAGS')
    for z in (before,after):
        need(z['host']==supervisor['host'] and z['boot_id']==supervisor['boot_id']
             and z['supervisor_PID']==supervisor['PID'] and z['scope_request_id']==supervisor['scope_request_id']
             and z['exe_identity']=='UNKNOWN' and z['executable_identity_verified'] is False,'TRANSPORT_CURRENT_SUPERVISOR')
        chain=z['chain'];need(len(chain)==4 and len({x['PID'] for x in chain})==4,'EXACT_TRANSPORT_CHAIN')
        need(chain[0]['PID']==supervisor['PID'] and chain[0]['startticks']==supervisor['startticks']
             and chain[0]['proc_uid']==supervisor['uid']==1001,'CURRENT_SUPERVISOR_STARTTICKS')
        need(all(chain[i]['PPID']==chain[i+1]['PID'] for i in range(3)) and chain[3]['PPID']==0
             and [x['proc_uid'] for x in chain]==[1001,1001,0,0],'DIRECT_PARENT_ROOT_CHAIN')
        for x in chain:
            s=x['status_identity']
            need(type(x['startticks'])is int and x['startticks']>0 and s['PPid']==x['PPID']
                 and s['Uid']==[x['proc_uid']]*4 and s['NSpid']==[x['PID']] and s['TracerPid']==0
                 and x['exe_identity']=='UNKNOWN' and x['exe_read'] is False and x['FD_targets_read'] is False,
                 'TRANSPORT_METADATA_IDENTITY_ONLY')
            need(all(state not in ('Z','X') and len(state)==1 for state in x['observed_states']), 'TRANSPORT_LIVE_OBSERVATION')
        need(all(x['comm']=='sshd' and x['fd_uid']==0 and x['fd_mode']==0o500 for x in chain[1:]),'ROOT_OWNED_TRANSPORT_FD_DIRECTORY')
        need(chain[1]['status_identity']['CapEff']=='0000000000000000' and chain[1]['status_identity']['NoNewPrivs']==0,
             'DIRECT_TRANSPORT_OBSERVED_STATUS')
    keys=('PID','PPID','PGID','startticks','comm','proc_uid','proc_device','proc_inode','fd_uid','fd_mode','fd_device','fd_inode','status_identity')
    need(all(all(x[k]==y[k] for k in keys) for x,y in zip(before['chain'],after['chain'])),'STABLE_TRANSPORT_CHAIN_METADATA')
    x=f['uninspected_transport_ancestor'];parent=before['chain'][1]
    need(all(x[k]==parent[k] for k in ('PID','PPID','startticks','proc_uid')) and isinstance(x['reason'],str) and x['reason'],
         'ONLY_CURRENT_DIRECT_PARENT_EXEMPT')
    return dict(transport_exception_count=1,all_UID_FD_complete=False,nontransport_UID_FD_complete=True,
                transport_FDs_inspected=False,exe_identity='UNKNOWN',exclusive_GPU_qualified=False,
                scope_policy='scope_admission_satisfied_with_transport_exception',uninspected_parent_PID=x['PID'],
                uninspected_parent_startticks=x['startticks'])


def audit(bundle,context):
    files=bundle['files'];errors=[];missing=[]
    result=dict(context=context,original_execution_state=bundle['original_execution_state'],
        planned_generations=30,arms={a:dict(denominator=10,completed=0,missing=10,verified=0) for a in ARMS},
        rows=[dict(arm=a,ordinal=i,state='UNOBSERVED',natural_EOS=None,cap_hit=None) for a in ARMS for i in range(10)],
        engineering_metadata_verified=False,independent_EOS_metadata_verified=False,
        gold_read=False,scored=False,accuracy=None,state_body_reads=0,Torch_imports=0,device_queries=0,
        original_domain_FAILURE_preserved=True,original_30_or_Model_replay_allowed=False,
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
                 and o['native_returncode'] is None and o['planned_generations']==30
                 and o['deadline']==context['epoch']+900,'FAILED_BEFORE_RELEASE')
            return True
        result.update(independent_failure_metadata_verified=bool(check('failed_admission',failure_gate)),
            post_status='UNKNOWN',predictions_observed=0,predictions_scored=0,errors=errors,missing_files=missing,
            prediction_read_gate='NOT_RELEASED_NO_PREDICTIONS',all_UID_FD_complete=False,
            exclusive_GPU_qualified=False,old_v1_v2_FAILURE_preserved=True)
        return result
    outcome=get('run/outcome.json');install=get('install-intent.json')
    mf=get('src/SOURCE_FILES.json');plan=get('src/core/PLAN.json');outer=get('src/INSTALL_PLAN.json')
    final=get('run/native/generation-finalized.json');generation=get('run/native/generation-plan.json')
    admission=get('run/worker-admission.json');owned=get('run/child-owner.json')

    def source():
        need(outcome['request_id']==install['request_id']==context['request_id']
             and outcome['publication']==install['publication']==context['publication']
             and outcome['epoch']==install['epoch']==context['epoch']
             and outcome['deadline']==install['deadline']==context['epoch']+900
             and outcome['finished_epoch']<=outcome['deadline'], 'ORIGINAL_CONTEXT')
        need(files['src/SOURCE_FILES.json']['SHA256']==context['source_manifest_SHA256']==outcome['source_manifest_SHA256'], 'ACTUAL_SOURCE_MF')
        for n in ('CLAIM.json','INSTALL_PLAN.json','core/SOURCE_FILES.json','core/PLAN.json','core/CLAIM.json'):
            need(files['src/'+n]['bytes']==mf['files'][n]['bytes'] and files['src/'+n]['SHA256']==mf['files'][n]['SHA256'], 'PUBLISHED_SOURCE_MEMBERS')
        need(mf['files']['core/infer_dev10.py']['SHA256']=='91d7424e29a1670a92dd5866ff6ed0b23cf7787e1dc403b74be940bf451888f3',
             'EXACT_REVIEWED_INFERENCE_CALLER')
        need(outer['core_manifest_SHA256']==files['src/core/SOURCE_FILES.json']['SHA256']
             and mf['files']['qualified/bounded_GPU_supervisor.py']['SHA256']=='2307123f57756f3a7a42153176b2e3acb068f6e0a51e5756730468236bbf0e2f', 'CORE_RSS_BINDING')
        need(admission['request_id']==context['request_id'] and admission['epoch']==context['epoch']
             and admission['deadline']==context['epoch']+820 and admission['whole_deadline']==context['epoch']+900
             and admission['owned_child']==owned['child'] and admission['supervisor']==owned['supervisor']
             and admission['source_manifest_SHA256']==outer['core_manifest_SHA256'], 'ACTUAL_HELD_ADMISSION')
        for g in (generation,final):
            need(g['arms']==list(ARMS) and g['dev_denominator_each']==10 and g['generation_cap']==256
                 and g['input_SHA256']==plan['dev_inputs']['SHA256']
                 and g['epoch']==context['epoch'] and g['deadline']==context['epoch']+820
                 and g['dev_gold_read'] is False and g['blind_inputs_gold_or_media_read'] is False
                 and g['domain_original_training_accepted'] is False, 'FIXED_THREE_ARM_INPUT_CONTRACT')
        pin=plan['native_identities'];ref=files['reference/native-identities']
        need(ref['bytes']==pin['bytes'] and ref['SHA256']==pin['SHA256'],'NATIVE_DEV_REFERENCE_PIN')
        refs=[x for x in ref['value'] if x['split']=='dev']
        need(len(refs)==10 and [x['ordinal'] for x in refs]==list(range(10)), 'NATIVE_TEN_REFERENCE')
        return refs
    refs=check('source_input_contract',source)

    def postgate():
        post=get('run/after.json');before=get('run/before.json');clean=outcome['cleanup']
        need(outcome['all_FD_clean_qualified'] is False and outcome['exclusive_GPU_proven'] is False
             and outcome['exe_identity']=='UNKNOWN' and outcome['scope_policy']==post['scope_policy']
             and outcome['transport_FDs_inspected'] is post['transport_FDs_inspected'],'OUTCOME_HONEST_SCOPE')
        need(post==outcome['post'] and post['active_compute_processes']==0 and post['master_port_bindable'] is True
             and len(post['devices'])==len(before['devices'])==1, 'ACTUAL_SCOPE_POST')
        result['FD_scope']={label:transport_scope(snapshot,owned['supervisor']) for label,snapshot in [('before',before),('after',post)]}
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
            rec=final['arm_results'][arm]
            need(rec['planned']==10 and 0<=rec['completed']<=rec['attempted']<=10 and rec['missing']==10-rec['completed'], 'ARM_DENOMINATOR')
            need(isinstance(values,list) and len(values)==rec['completed'], 'SAVED_PREDICTION_COUNT')
            if 'file' in rec:
                need(rec['file']==arm+'-predictions-private.jsonl','EXACT_ARM_FILE')
                row=files['run/native/'+rec['file']]
                need(row['bytes']==rec['bytes'] and row['SHA256']==rec['SHA256'], 'FINALIZED_PREDICTION_PIN')
            need([x['ordinal'] for x in values]==list(range(len(values))), 'ORDERED_UNIQUE_SLOTS')
            return True
        ok=check(arm+'-file',armgate)
        count=len(values) if isinstance(values,list) and len(values)<=10 else 0
        result['arms'][arm].update(completed=count,missing=10-count,input_SHA256=plan['dev_inputs']['SHA256'] if plan else None)
        for ordinal in range(10):
            row=result['rows'][ARMS.index(arm)*10+ordinal];row['state']='MISSING'
            returned=get('run/native/'+arm+'-native-return-%02d.json'%ordinal,False)
            row['native_return_present']=returned is not None
            if not ok or ordinal>=count:
                continue
            def slot():
                x=values[ordinal];need(refs is not None,'INPUT_REFERENCE_UNQUALIFIED')
                need(x['arm']==arm and x['status']=='COMPLETED' and x['ordinal']==ordinal
                     and H(x['id'].encode())==refs[ordinal]['source_id_SHA256']
                     and x['native_field_identity_verified'] is True,'ACTUAL_INPUT_SLOT')
                need(returned is not None and returned['actual_native_infer_returned'] is True
                     and returned['qualified_prediction'] is False and returned['arm']==arm
                     and returned['ordinal']==ordinal and returned['id']==x['id']
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
    result.update(old_v1_v2_FAILURE_preserved=True,exclusive_GPU_qualified=False,
        errors=errors,missing_files=sorted(set(missing)),verified_predictions=verified_total,
        independent_EOS_metadata_verified=verified_total==30,
        engineering_metadata_verified=bool(not errors and not missing and verified_total==30 and clean
            and outcome['engineering_completed'] is True and outcome['native_returncode']==0
            and not outcome['cleanup']['termination_needed'] and final['passed'] is True
            and final['all_predictions_finalized_before_gold'] is True
            and final['all_prediction_files_closed_before_gold'] is True),
        input_tensor_identity_scope='Saved slot IDs + published original native45/dev10 pin + frozen inference field-equality guard; no independent tensor re-encoding.',
        adapter_scope='Frozen caller requires actual selected504 torch.equal + same-base identity; no new adapter body read.',
        score_next='Only a separately authorized original score_after process may read gold after full30 prediction seal; this reader never scores.',
        original_small_files={n:{k:r[k] for k in ('bytes','SHA256','exists','error_type') if k in r} for n,r in files.items()})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('bundle','bundle-sha256','context','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();b=Path(a.bundle);need(b.stat().st_size<=8*1024**2,'BUNDLE8M')
    raw=b.read_bytes();need(H(raw)==a.bundle_sha256,'ORIGINAL_PRIVATE_BUNDLE_SHA')
    bundle=json.loads(raw);context=json.loads(Path(a.context).read_bytes())
    need(context['namespace']=='/home/whj/nvidia-baseline/alignment-assets/training-USGS-T04-dev10-effect-A100-launch-v3','V3_EXACT_NAMESPACE')
    result=audit(bundle,context)
    with Path(a.output).open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result.get(k) for k in ('original_execution_state','planned_generations','verified_predictions',
          'engineering_metadata_verified','gold_read','effect_qualified')}))
