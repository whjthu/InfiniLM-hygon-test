"""One offline metadata-only audit of the original control45 CPU receipt; no native replay."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

G=Path(__file__).resolve().parents[3]
SOURCE=G/'code/bw1000/training-USGS-T04-generic-control-native-CPU-v1'
BUNDLE=G/'local-only/training-USGS-T04-generic-control-native-CPU-v1-independent-original/stdout.private'
RECEIPT=G/'local-only/training-USGS-T04-generic-control-native-CPU-v1-original/receipt.json'
ROSTER=G/'local-only/training-USGS-T04-domain-control-adapter-v1-preparation/generic45-roster-private.json'
OUT=G/'local-only/training-USGS-T04-generic-control45-native-root-audit-v1/final-metadata'


def need(ok,why):
    if not ok:raise ValueError(why)

def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
def read(p):
    need(p.is_file() and not p.is_symlink(),'Regular pinned file required');a=p.stat();raw=p.read_bytes();b=p.stat();need(a.st_size==b.st_size==len(raw) and a.st_mtime_ns==b.st_mtime_ns,'File unstable');return raw

def audit():
    OUT.mkdir();raw=read(BUNDLE);need(sha(raw)=='c42ee639ed3e3babc56c754ea21540ff7e0542ac41121ab59d93c6a95cc14c31','Original bundle differs')
    bundle=json.loads(raw);f=bundle['files'];logs={'run/controller/worker.stdout','run/controller/worker.stderr'}
    expected={'install-intent.json','run/controller/intent.json','run/controller/outcome.json','run/controller/post.json','run/controller/child-owner.json','run/controller/held-child-anchor.json','run/controller/owned-registry.json','run/controller/worker.stdout','run/controller/worker.stderr','run/native/result.json','run/native/progress.json','run/native/native-identities-private.json','run/native/processor-provider-private.json','data/data-binding-private.json','data/control-provenance-private.json','data/prepare-progress.json'}
    need(set(f)==expected,'Only original16 metadata receipts admitted; no raw caption/input/weights members')
    files={}
    for name,e in f.items():
        need(e['exists'] is True and type(e['bytes']) is int and e['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',e['SHA256']),'Receipt member metadata differs')
        files[name]=dict(bytes=e['bytes'],SHA256=e['SHA256'])
        if name in logs:continue # Never inspect log bodies.
        value=e['value'];serialized=canonical(value)
        if name=='install-intent.json':serialized=(json.dumps(value)+'\n').encode()
        need(len(serialized)==e['bytes'] and sha(serialized)==e['SHA256'],'JSON metadata cannot reproduce pinned source serialization')
    receipt_raw=read(RECEIPT);receipt=json.loads(receipt_raw);install=f['install-intent.json']['value'];intent=f['run/controller/intent.json']['value'];outcome=f['run/controller/outcome.json']['value'];native=f['run/native/result.json']['value'];progress=f['run/native/progress.json']['value']
    need(receipt['passed'] is True and receipt['rc']==0 and receipt['exec_request_attempted'] is True,'Original request not successful known execution')
    for x in [receipt,install,intent,outcome]:
        need(x['request_id']=='26fa3a953d3d40caa6639d035ac274b9' and x['publication']=='de15d9b6d09051c0dcbc3ff2e0ffb4cf3736e578'
            and x['epoch']==1794951686.4229157 and x['deadline']==1794951986.4229157,'Original context differs')
    mf=read(SOURCE/'SOURCE_FILES.json');need(sha(mf)==install['source_manifest_SHA256']==intent['source_manifest_SHA256'],'Actual source MF differs')
    for name,pin in json.loads(mf)['files'].items():
        b=read(SOURCE/name);need(len(b)==pin['bytes'] and sha(b)==pin['SHA256'],'Source pin differs')
    need(sha(read(SOURCE/'CLAIM.json'))==install['claim_SHA256'],'Claim differs')
    plan=json.loads(read(SOURCE/'PLAN.json'))
    offsets={'deadline':300,'work_deadline':260,'supervisor_stop':270,'held_cleanup_deadline':282,'registry_cleanup_deadline':285,'alarm_deadline':290,'client_deadline':295}
    for key,offset in offsets.items():need(intent[key]==outcome[key]==intent['epoch']+offset,'Single-origin CPU clock differs')
    need(native['epoch']==intent['epoch'] and native['deadline']==intent['work_deadline'] and outcome['finished']<intent['supervisor_stop'],'Native epoch reset or terminal late')
    need(outcome['passed'] is True and outcome['worker_returncode']==0 and outcome['failure'] is None and outcome['stopped'] is False,'CPU supervisor terminal failed')
    need(intent['CPU_affinity']==[2,3] and intent['GPU']==outcome['GPU']==outcome['actual_updates']==0 and intent['RSS_stop_threshold_bytes']==8589934592
        and intent['RSS_is_hard_limit'] is False and outcome['RSS_is_hard_limit'] is False and outcome['observed_group_RSS_peak_bytes']<8589934592,'Actual CPU/RSS resource differs')
    owner=intent['owner'];need(owner==outcome['owner'] and owner['uid']==install['UID'] and owner['host']==install['host'] and owner['boot_id']==install['boot_id'],'Owner identity differs')
    child=f['run/controller/child-owner.json']['value'];anchor=f['run/controller/held-child-anchor.json']['value'];registry=f['run/controller/owned-registry.json']['value']
    need(anchor['supervisor']==owner and anchor['request_id']==receipt['request_id'] and anchor['start_new_session'] is True
        and anchor['PID']==anchor['PGID']==child['PID']==child['PGID'] and child['PPID']==owner['PID'] and registry[str(child['PID'])]==child,'Held child/registry ownership differs')
    post=f['run/controller/post.json']['value'];need(post==outcome['post'] and post['same_host'] is True and post['same_boot'] is True and post['controller_device_FDs']==[]
        and post['held_child_reaped'] is True and post['owned_descendants_remaining']==0 and post['owned_device_FD_holders']=={},'Postcleanup not independent clean')
    need(outcome['held_cleanup']['PID']==child['PID'] and outcome['held_cleanup']['reaped'] is True and outcome['held_cleanup']['returncode']==0
        and outcome['held_cleanup']['termination_needed'] is False and outcome['held_cleanup']['owned_group_signaled'] is False
        and outcome['registry_cleanup']['owned_processes_remaining']==0 and outcome['registry_cleanup']['termination_needed'] is False,'Normal reap/cleanup differs')
    need(native['passed'] is True and native['planned_train']==native['completed_train']==45 and native['planned_dev']==native['completed_dev']==0
        and native['shared_dev_rows_reused']==10 and native['shared_dev_new_encodes']==0 and native['actual_control_images']==45 and native['actual_updates']==0,'New45/shared10 denominator differs')
    need(native['framework_files']==396 and native['model_constructed'] is False and native['actual_model_freeze_or_trainable_initialization_qualified'] is False
        and native['denied_attempts']==[] and native['raw_caption_exported'] is False and native['raw_processed_payload_saved'] is False
        and native['dev_gold_read'] is False and native['blind_data_read'] is False,'Forbidden model/source export or unqualified freeze claim')
    need(outcome['native_result']['SHA256']==files['run/native/result.json']['SHA256'] and outcome['native_result']['bytes']==files['run/native/result.json']['bytes'],'Supervisor native hash differs')
    need(progress['failed'] is False and progress['planned_train']==progress['completed_train']==45 and progress['planned_dev']==progress['completed_dev']==0,'Full progress denominator differs')
    db=f['data/data-binding-private.json']['value'];prov=f['data/control-provenance-private.json']['value'];prep=f['data/prepare-progress.json']['value'];roster_raw=read(ROSTER)
    need(sha(roster_raw)==plan['roster_SHA256']=='e44f53095e004c99898cdf97835791b96fc4496cc3b11a93bf2c4e8e7294d71b','Frozen pre-answer roster differs');roster=json.loads(roster_raw)
    need(native['data_binding_SHA256']==files['data/data-binding-private.json']['SHA256']=='042f1192d32bb50a0eea8072f9f66ad345a3fb277cdb77dfea62885d1d6f8ced'
        and db['provenance_SHA256']==files['data/control-provenance-private.json']['SHA256'] and db['files']==native['adapter_files'],'Data/provenance/adapter hashes differ')
    need(db['passed'] is True and db['generic_roster_SHA256']==prov['roster_SHA256']==native['fixed_roster_SHA256']==plan['roster_SHA256']
        and db['generic_original_dataset_SHA256']==prov['original_JSONL_SHA256']==native['original_generic_JSONL_SHA256']==plan['generic_original_dataset']['SHA256'],'Original64/full45 source chain differs')
    need(db['domain_native_qualification_SHA256']==native['shared_dev_qualification_SHA256']==plan['native_declared_qualification']['SHA256']=='24773954a6d0c5513aa6f3c06a93a5faa96bea64fc4c0ece46362358888f4850'
        and db['shared_dev_input_SHA256']==plan['shared_dev_reference']['dev_input']['SHA256'] and db['shared_dev_native_identities_SHA256']==plan['shared_dev_reference']['native_identities']['SHA256']
        and db['shared_dev_rows_reused']==10 and db['shared_dev_new_encodes']==0,'Shared dev only qualification reuse differs')
    need(all(db[k] is False for k in ['dev_gold_read','blind_inputs_gold_or_media_read','caption_payload_exported_to_PC','literal_special_tokens_added','native_qualified','training_GO','GPU_GO'])
        and prov['caption_payload_exported_to_PC'] is False and prov['blind_or_dev_gold_read'] is False and prov['original_question_and_caption_full'] is True and prov['response_prefix']=='FINAL: '
        and prov['selected_count']==45 and prov['source_count']==64 and prov['actual_GPU']==0,'Control format/export provenance differs')
    identities=f['run/native/native-identities-private.json']['value'];need(len(identities)==45 and native['identities_SHA256']==files['run/native/native-identities-private.json']['SHA256'],'Actual identity denominator differs')
    need([r['ordinal'] for r in identities]==list(range(45)) and all(r['split']=='train' for r in identities) and len({r['source_id_SHA256'] for r in identities})==45,'Native order/unique IDs differ')
    need(len(prov['selected'])==len(db['media'])==45 and sum(r['bytes'] for r in db['media'].values())==7321484 and prep['completed_images']==prep['planned_images']==45
        and prep['actual_image_bytes']==7321484 and prep['shared_dev_new_encodes']==0,'All45 media/progress denominator differs')
    for ordinal,(reference,p,record) in enumerate(zip(roster['selected'],prov['selected'],identities)):
        expected_id='generic-control45:'+str(reference['original_ordinal']).zfill(2)+':'+reference['image_SHA256'];idsha=sha(expected_id.encode())
        need(p['source_ordinal']==reference['original_ordinal'] and p['image_SHA256']==reference['image_SHA256'] and p['source_id_SHA256']==record['source_id_SHA256']==idsha,'Fixed ordered selected identity differs')
        need(set(p)=={'source_ordinal','source_id_SHA256','image_name','image_SHA256','original_answer_SHA256','transformed_answer_SHA256','FINAL_prefix_added'}
            and all(re.fullmatch('[0-9a-f]{64}',p[k]) for k in ['original_answer_SHA256','transformed_answer_SHA256']) and p['FINAL_prefix_added'] is True,'Hash-only full caption provenance schema differs')
        media=db['media'][p['image_name']];need(media['bytes']==reference['image_bytes'] and media['SHA256']==reference['image_SHA256'] and p['image_name'].startswith('train/media/'),'Image content metadata differs')
        need(record['full_input_retained'] is True and record['labels_None'] is False and 0<record['first_supervised_index']<record['full_length']<=4096
            and type(record['effective_supervised_tokens']) is int and record['effective_supervised_tokens']==record['full_length']-record['first_supervised_index']
            and record['supervised_EOS_count']==1 and record['native_EOS_ids'] and all(type(i) is int for i in record['native_EOS_ids']),'Native full supervision/mask/EOS/token counter metadata differs')
        for collection in ['encoded_fields','collated_fields']:
            need({'input_ids','labels','pixel_values','image_grid_thw'}<=set(record[collection]),'Required native fields missing')
            for field,v in record[collection].items():need(v['finite'] is True and type(v['elements']) is int and v['elements']>=0 and re.fullmatch('[0-9a-f]{64}',v['SHA256']),'Native field fingerprint invalid')
            if collection=='collated_fields':
                need(record[collection]['input_ids']['elements']==record[collection]['labels']['elements']==record['full_length'],'Actual collated full input/label lengths differ')
            else:
                need(record[collection]['input_ids']['elements']==record[collection]['labels']['elements']==0 and record[collection]['input_ids']['tensors']==record[collection]['labels']['tensors']==0,'Encoded Python-list fingerprint ABI differs')
            need(record[collection]['pixel_values']['elements']>0,'Actual native vision tensor is empty')
        need(record['collated_fields']['attention_mask']['elements']==record['full_length'],'Full attention mask length differs')
    tokens=[r['effective_supervised_tokens'] for r in identities];lengths=[r['full_length'] for r in identities]
    need(sum(tokens)==native['effective_supervised_token_total']==694 and [min(tokens),max(tokens)]==native['effective_supervised_token_range']==[12,20]
        and [min(lengths),max(lengths)]==native['lengths']['train']==[255,381],'Actual full45 supervision/length totals differ')
    provider=f['run/native/processor-provider-private.json']['value'];need(files['run/native/processor-provider-private.json']['SHA256']=='37a27873338dc71addf59549ec23bf3b9a9ecbc17be301dcbb0102817dfc8877'
        and provider['wrapper_order']==['outer','retry','implementation'] and len(provider['wrapper_chain'])==3 and provider['retry_count']==3
        and provider['closure_targets_match_wrapped'] is True and provider['invocation_preserves_decorators'] is True and provider['unwrapped_leaf_called'] is False,'Actual original wrapper receipt differs')
    result=dict(state='USGS_T04_GENERIC_CONTROL45_NATIVE_CPU_INDEPENDENT_PASS_NO_MODEL_TRAINING',passed=True,case='T04.sft',request_id=receipt['request_id'],source_publication=receipt['publication'],
        epoch=intent['epoch'],deadline=intent['deadline'],finished=outcome['finished'],new_native_planned=45,new_native_completed=45,new_dev_completed=0,shared_dev_qualified_rows_reused=10,shared_dev_new_encodes=0,
        failed_rows=0,missing_rows=0,fixed_ordered_roster_verified=45,original_generic_source_rows=64,selected_generic_source_rows=45,selected_images=45,selected_image_bytes=7321484,
        original_generic_JSONL_SHA256=plan['generic_original_dataset']['SHA256'],roster_SHA256=plan['roster_SHA256'],canonical45_input= db['files']['train-swift.jsonl'],data_binding_SHA256=native['data_binding_SHA256'],
        native_identities_SHA256=native['identities_SHA256'],control_provenance_SHA256=files['data/control-provenance-private.json']['SHA256'],ordered_source_ids_verified=45,
        full_input_retained=True,max_length=4096,max_pixels=262144,actual_length_range=[255,381],effective_shifted_supervised_tokens_unique45=694,effective_shifted_supervised_token_range=[12,20],
        token_counter_consistency='Sum all45 producer counters; each equals full_length-first_supervised_index; source formula uses shifted nonignore labels and actual attention mask. No raw labels were reread.',
        train_labels_masks_native_EOS_verified=True,supervised_EOS_count=45,EOS_identity_SHA256=sha(canonical([r['native_EOS_ids'] for r in identities])),
        wrapper_order=provider['wrapper_order'],actual_decorated_encode=True,unwrapped_leaf_called=False,wrapper_receipt_SHA256=files['run/native/processor-provider-private.json']['SHA256'],
        retry_policy_count=3,actual_retry_attempts=None,actual_retry_attempts_evidence='Not instrumented; do not equate decorator policy with measured execution count',
        explicit_FINAL_format_changed=True,full_caption_preservation_basis='Pinned original64 source/transformation code plus hash-only provenance; no second caption body read; not an independent semantic caption audit',
        raw_caption_body_read_in_audit=False,reader_log_bodies_not_inspected=True,first_reader_scalar_fingerprint_rejection_preserved=True,raw_processed_payload_read=False,weight_reads=0,source_image_bytes_reread_in_audit=0,new_remote_calls=0,new_native_calls=0,
        shared_dev_qualification_SHA256=plan['native_declared_qualification']['SHA256'],CPU_affinity=[2,3],CPU_wall_seconds=native['CPU_wall_seconds'],whole_scope_seconds=outcome['finished']-intent['epoch'],
        peak_self_RSS_bytes=native['peak_self_RSS_bytes'],sampled_group_RSS_peak_bytes=outcome['observed_group_RSS_peak_bytes'],RSS_samples=outcome['RSS_samples'],RSS_stop_threshold_bytes=8589934592,RSS_is_hard_limit=False,
        original_epoch_never_reset=True,controller_PID=owner['PID'],controller_startticks=owner['startticks'],worker_PID=child['PID'],worker_startticks=child['startticks'],
        normal_exit_and_owned_postclean=True,controller_device_FDs=0,owned_descendants_remaining=0,held_child_reaped=True,termination_needed=False,
        actual_GPU_cards=0,actual_model_constructed=False,actual_updates=0,framework_files=396,actual_model_freeze_qualified=False,initial_trainable_values_qualified=False,
        native_preprocessing_qualified=True,Dataset_GO=False,training_GO=False,GPU_GO=False,Hygon_GO=False,Model_GO=False,formal_performance=False,heldout_effect=False,
        original16_bundle=dict(path=BUNDLE.relative_to(G).as_posix(),bytes=len(raw),SHA256=sha(raw),member_count=16,JSON_metadata_members_checked=14,log_bodies_ignored=2),
        original_invocation_receipt=dict(path=RECEIPT.relative_to(G).as_posix(),bytes=len(receipt_raw),SHA256=sha(receipt_raw)),original_metadata_files=files,
        source_manifest_SHA256=sha(mf),claim_SHA256=install['claim_SHA256'],reader_SHA256=sha(Path(__file__).read_bytes()))
    (OUT/'result.json').write_bytes(canonical(result));print(json.dumps({k:result[k] for k in ['passed','new_native_completed','new_dev_completed','shared_dev_qualified_rows_reused','effective_shifted_supervised_tokens_unique45','native_identities_SHA256','data_binding_SHA256']},sort_keys=True));return result

if __name__=='__main__':audit()
