"""One independent metadata-only terminal read; no inputs, PNGs, labels or native import."""
import argparse,json,os,socket,time
from pathlib import Path
from common import D,need,pinned,sha,canonical,verify_source


def read(scope,request,publication,epoch):
    verify_source();plan=json.loads((D/'PLAN.json').read_bytes());pins={}
    def load(path,cap=2*1024**2):
        before=path.lstat();need(path.is_file() and not path.is_symlink() and 0<before.st_size<=cap,'BOUNDED_METADATA')
        raw=path.read_bytes();after=path.lstat()
        need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'STABLE_METADATA')
        pins[str(path.relative_to(scope))]=dict(bytes=len(raw),SHA256=sha(raw));return json.loads(raw)
    install=load(scope/'install-intent.json');intent=load(scope/'run/controller/intent.json')
    out=load(scope/'run/controller/outcome.json');post=load(scope/'run/controller/post.json') if (scope/'run/controller/post.json').exists() else None
    source_sha=sha((D/'SOURCE_FILES.json').read_bytes());claim_sha=sha((D/'CLAIM.json').read_bytes())
    need(install['source_manifest_SHA256']==intent['source_manifest_SHA256']==source_sha
         and install['claim_SHA256']==claim_sha,'ORIGINAL_PUBLISHED_SOURCE_AND_CLAIM')
    for row in [install,intent,out]:
        need(row['request_id']==request and row['publication']==publication and row['epoch']==epoch and row['deadline']==epoch+300,'ONE_ORIGINAL300_CONTEXT')
    owner=out['owner'];need(owner==intent['owner'] and owner['PID']==install['PID'] and owner['startticks']==install['startticks']
          and owner['host']==socket.gethostname() and owner['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'STRONG_OWNER_SAME_HOST_BOOT')
    proc=Path('/proc')/str(owner['PID'])
    try:
        fields=(proc/'stat').read_text().split(') ',1)[1].split()
        need(int(fields[19])!=owner['startticks'],'ORIGINAL_CONTROLLER_STILL_ALIVE')
        owner_state='ORIGINAL_IDENTITY_ABSENT_PID_REUSED'
    except (FileNotFoundError,ProcessLookupError):owner_state='ORIGINAL_PID_ABSENT'
    progress=load(scope/'run/native/progress.json') if (scope/'run/native/progress.json').exists() else None
    native=load(scope/'run/native/result.json') if (scope/'run/native/result.json').exists() else None
    if native is not None:
        need(out['native_result'] is not None
             and pins['run/native/result.json']['bytes']==out['native_result']['bytes']
             and pins['run/native/result.json']['SHA256']==out['native_result']['SHA256'],'ORIGINAL_NATIVE_RESULT_PIN')
    result=dict(request_id=request,publication=publication,epoch=epoch,deadline=epoch+300,planned_blind=25,completed_blind=progress['completed_blind'] if progress else 0,
        state='KNOWN_CPU_FAILURE_FULL25_DENOMINATOR',producer_passed=out['passed'],independent_passed=False,owner_terminal_state=owner_state,
        blind_gold_read=False,train_or_dev_read=False,new_processor_calls=0,actual_updates=0,GPU_GO=False,training_GO=False,Model_GO=False)
    if out['passed'] is True:
        need(out['worker_returncode']==0 and out['failure'] is None and out['stopped'] is False and out['cleanup_errors']=={},'NORMAL_CPU_WORKER_REQUIRED')
        need(out['held_cleanup']['reaped'] is True and out['held_cleanup']['termination_needed'] is False
             and out['registry_cleanup']['owned_processes_remaining']==0 and out['registry_cleanup']['termination_needed'] is False,'OWNED_CLEANUP')
        need(post==out['post'] and post['same_host'] is True and post['same_boot'] is True and post['controller_device_FDs']==[]
             and post['owned_descendants_remaining']==0 and post['owned_device_FD_holders']=={},'CPU_OWNED_POST')
        need(out['resources']['soft_limit_bytes']==8*1024**3 and out['RSS_is_hard_limit'] is False and out['RSS_unknown_not_zero'] is True
             and out['resources']['soft_limit_exceeded'] is False and out['resources']['samples']>0,'QUALIFIED_RSS_SAMPLED_SOFT_LIMIT')
        need(native['passed'] is True and native['completed_blind']==25 and native['planned_blind']==25 and native['epoch']==epoch
             and native['deadline']==epoch+260 and native['actual_mode']=='transformers' and native['denied_attempts']==[]
             and native['actual_updates']==0 and native['model_constructed'] is False and native['blind_gold_read'] is False
             and native['train_or_dev_data_read'] is False and native['raw_processed_payload_saved'] is False,'USER_ONLY_NATIVE_COMPLETION')
        need(native['Swift_sources_verified']==7 and native['extra_wrapper_sources_verified']==3 and native['actual_versions']==plan['versions'] and native['metadata_only_files']==14
             and native['before_device_FDs']==native['after_device_FDs']==[],'NO_MODEL_DEVICE_PROVIDER_SCOPE')
        need(progress['completed_blind']==25 and progress['planned_blind']==25 and progress['failed'] is False,'ALL25_PROGRESS')
        data=load(scope/'data/data-binding-private.json');rows=load(scope/'run/native/native-identities-private.json');provider=load(scope/'run/native/processor-provider-private.json')
        need(data['blind_gold_read'] is False and data['train_or_dev_data_read'] is False and data['media']==plan['media']
             and data['dataset_manifest_SHA256']==plan['dataset_manifest']['SHA256'] and data['source_inputs']==plan['blind_inputs'],'FROZEN_SOURCE_BINDING')
        need(len(rows)==len(data['ordered_source_id_SHA256'])==25 and len(set(data['ordered_source_id_SHA256']))==25,'ALL25_IDENTITIES')
        for i,row in enumerate(rows):
            need(row['ordinal']==i and row['split']=='blind' and row['source_id_SHA256']==data['ordered_source_id_SHA256'][i]
                 and row['labels_None'] is True and row['full_input_retained'] is True and row['supervised_EOS_count'] is None
                 and row['first_supervised_index'] is None and type(row['full_length']) is int and 0<row['full_length']<=4096,'NO_SUPERVISION_TRUNCATION')
            for fields in [row['encoded_fields'],row['collated_fields']]:
                need('input_ids' in fields and 'pixel_values' in fields and all(x['finite'] is True for x in fields.values()),'FINITE_NATIVE_FIELDS')
            need(row['native_EOS_ids'] and all(type(x) is int for x in row['native_EOS_ids']),'TOKENIZER_EOS_IDENTITY_ONLY')
        policy=json.loads((D/'WRAPPER_SOURCES.json').read_bytes())
        need(provider['binding_kind']=='EXACT_DECORATED_NATIVE_ENCODE_CHAIN' and provider['closure_targets_match_wrapped'] is True
             and provider['invocation_preserves_decorators'] is True and provider['unwrapped_leaf_called'] is False
             and len(provider['wrapper_chain'])==3 and provider['retry_count']==3,'ACTUAL_NATIVE_WRAPPER_CHAIN')
        for actual,expected in zip(provider['wrapper_chain'],policy['chain']):
            need(actual['path']==expected['path'] and actual['code_filename']==expected['path'] and actual['inspect_sourcefile']==expected['path']
                 and actual['bytes']==expected['bytes'] and actual['file_SHA256']==expected['SHA256']
                 and actual['code_qualname']==expected['code_qualname'] and actual['role']==expected['role'],'CLOSED_WRAPPER_PROVIDER')
        need(pins['run/native/native-identities-private.json']['SHA256']==native['identities_SHA256'],'PRODUCER_IDENTITIES_PIN')
        result.update(independent_passed=True,state='BLIND25_USER_ONLY_NATIVE_CPU_QUALIFIED',completed_blind=25,
            token_length_range=[min(r['full_length'] for r in rows),max(r['full_length'] for r in rows)],
            labels_None=25,finite_fields=25,full_inputs_retained=25,unique_media=5,closure_groups=3,
            sampled_group_peak_RSS_bytes=out['observed_group_RSS_peak_bytes'],CPU_wall_seconds=native['CPU_wall_seconds'])
    else:
        need(out['worker_returncode'] is not None and out['finished']>=epoch,'KNOWN_TERMINAL_FAILURE')
        result.update(independent_passed=True,producer_failure_retained=True,missing_blind=25-result['completed_blind'])
    result['original_metadata_files']=pins;return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scope',type=Path,required=True);p.add_argument('--request-id',required=True)
    p.add_argument('--publication',required=True);p.add_argument('--epoch',type=float,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    need(not a.output.exists(),'NEW_PRIVATE_READER_OUTPUT');r=read(a.scope,a.request_id,a.publication,a.epoch)
    a.output.write_bytes(canonical(r));print(json.dumps(r));raise SystemExit(0 if r['independent_passed'] else 1)
