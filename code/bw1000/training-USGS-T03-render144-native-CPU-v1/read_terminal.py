"""Pre-registered metadata-only consumer. Never opens any input, PNG, label or tensor."""
import argparse,json,math,os,signal,socket,time
from pathlib import Path
from prepare_inputs import D,canonical,digest,need,pinned
def read(a):
    reader_end=min(a.epoch+295,a.reader_origin+35);need(math.isfinite(a.reader_origin) and a.epoch<=a.reader_origin<=time.time()<reader_end,'READER_SAME_ORIGINAL300')
    started=time.monotonic();left=reader_end-time.time()
    def check():need(min(reader_end-time.time(),left-(time.monotonic()-started))>0,'READER_DEADLINE')
    def alarm(*_):raise TimeoutError('READER_DEADLINE')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,left)
    required=set([2, 3]);need(required<=os.sched_getaffinity(0),'READER_FIXED2CPU');os.sched_setaffinity(0,required)
    pins={};total=0
    def load(path,key,cap=4*1024**2,expected=None):
        nonlocal total
        check();s=path.lstat();need(path.is_file() and not path.is_symlink() and 0<s.st_size<=cap,'BOUNDED_METADATA')
        with path.open('rb') as f:
            stamp=lambda q:(q.st_dev,q.st_ino,q.st_size,q.st_mtime_ns,q.st_ctime_ns)
            need(stamp(s)==stamp(os.fstat(f.fileno())),'METADATA_OPEN_IDENTITY')
            raw=f.read(s.st_size+1);total+=len(raw);need(total<=16*1024**2,'METADATA_AGGREGATE16M')
            need(len(raw)==s.st_size and stamp(s)==stamp(os.fstat(f.fileno()))==stamp(path.lstat()),'METADATA_STABLE_FD')
        pin=dict(bytes=len(raw),SHA256=digest(raw));pins[key]=pin
        if expected:need(pin['bytes']==expected['bytes'] and pin['SHA256']==expected['SHA256'],'ORIGINAL_METADATA_PIN')
        return raw
    def obj(name,required=True):
        p=a.scope/name
        if not p.exists():
            if required:raise ValueError('REQUIRED_METADATA_ABSENT:'+name)
            return None
        return json.loads(load(p,name))
    mfraw=load(D/'SOURCE_FILES.json','source/SOURCE_FILES.json');need(digest(mfraw)==a.source_manifest_sha256,'ROOT_EXPECTED_SOURCE_MF')
    mf=json.loads(mfraw)['files'];source={}
    for name,pin in mf.items():
        need(Path(name).name==name,'SOURCE_NAME');source[name]=load(D/name,'source/'+name,2*1024**2,pin)
    need(digest(source['CLAIM.json'])==a.claim_sha256,'ROOT_EXPECTED_CLAIM')
    claim=json.loads(source['CLAIM.json']);plan=json.loads(source['PLAN.json'])
    need(claim['root_GO'] is True and claim['READ_GO'] is True,'ROOT_METADATA_READ_ADOPTION')
    need(str(a.scope)==plan['namespace'] and D==a.scope/'src','EXACT_ORIGINAL_NAMESPACE')
    install=obj('install-intent.json');intent=obj('run/controller/intent.json',False);out=obj('run/controller/outcome.json',False)
    need(install['request_id']==a.request_id and install['publication']==a.publication and install['epoch']==a.epoch and install['deadline']==a.epoch+300 and install['source_manifest_SHA256']==a.source_manifest_sha256 and install['claim_SHA256']==a.claim_sha256,'ORIGINAL_INSTALL_CONTEXT')
    result=dict(request_id=a.request_id,publication=a.publication,epoch=a.epoch,deadline=a.epoch+300,planned=plan['planned'],completed={s:0 for s in plan['planned']},missing=dict(plan['planned']),metadata_verified=False,native_preprocessing_qualified=False,process_terminal='UNKNOWN',GPU_GO=False,training_GO=False,Model_GO=False,model_freeze_or_initialization_qualified=False,tensor_math_independently_recomputed=False,new_native_calls=0,reader_inputs_or_media_read=False,reader_labels_or_gold_read=False,replay_allowed=False)
    def gone(owner):
        need(owner['host']==socket.gethostname() and owner['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'ORIGINAL_PROCESS_OTHER_HOST_BOOT_UNKNOWN')
        try:
            p=Path('/proc')/str(owner['PID'])/'stat';raw=p.read_bytes();fields=raw.split(b') ',1)[1].split()
            need(int(fields[19])!=owner['startticks'] or fields[0]==b'Z','ORIGINAL_PROCESS_NOT_TERMINAL')
        except (FileNotFoundError,ProcessLookupError):pass
    installer_owner=dict(PID=install['PID'],startticks=install['startticks'],host=install['host'],boot_id=install['boot_id'])
    gone(installer_owner);result['process_terminal']='SAME_HOST_BOOT_ORIGINAL_PID_ABSENT_REUSED_OR_Z'
    if intent is None or out is None:
        failure=obj('install-failure.json',False);result.update(state='INCOMPLETE_CONTROLLER_METADATA',install_failure_type=failure.get('error_type') if failure else None,post='UNKNOWN')
        return finish(result,pins,total)
    for row in (intent,out):
        need(row['request_id']==a.request_id and row['publication']==a.publication and row['epoch']==a.epoch and row['deadline']==a.epoch+300,'SAME_ORIGINAL300_CONTEXT')
    need(intent['source_manifest_SHA256']==a.source_manifest_sha256 and intent['owner']==out['owner'],'OWNER_SOURCE_BINDING')
    owner=out['owner'];need(owner['PID']==install['PID'] and owner['startticks']==install['startticks'] and owner['uid']==install['UID'],'EXEC_SAME_PROCESS_IDENTITY')
    gone(owner)
    child=obj('run/controller/child-owner.json',False);held=obj('run/controller/held-child-anchor.json',False)
    if child:
        need(held and child['PID']==held['PID'] and child['PGID']==held['PGID'] and held['request_id']==a.request_id and held['supervisor']==owner,'HELD_CHILD_ORIGINAL_IDENTITY')
        gone(dict(child,host=owner['host'],boot_id=owner['boot_id']))
    post=obj('run/controller/post.json',False);progress=obj('run/native/progress.json',False);native=obj('run/native/result.json',False)
    if progress:
        for s in plan['planned']:
            n=progress['completed_'+s];need(type(n) is int and 0<=n<=plan['planned'][s] and progress['planned_'+s]==plan['planned'][s],'FULL144_DENOMINATOR')
            result['completed'][s]=n;result['missing'][s]=plan['planned'][s]-n
    if native:
        need(out['native_result'] and pins['run/native/result.json']=={k:out['native_result'][k] for k in ('bytes','SHA256')},'NATIVE_RESULT_ORIGINAL_PIN')
    if out['passed'] is not True:
        result.update(metadata_verified=True,state='KNOWN_CPU_FAILURE_ORIGINAL144_DENOMINATOR',failure_type=(out.get('failure') or {}).get('type'),worker_returncode=out['worker_returncode'],producer_failure_retained=True,post=post)
        return finish(result,pins,total)
    need(child and out['worker_returncode']==0 and out['failure'] is None and not out['stopped'] and out['cleanup_errors']=={},'NORMAL_WORKER_RETURN')
    need(out['held_cleanup']['reaped'] is True and out['held_cleanup']['termination_needed'] is False and out['registry_cleanup']['owned_processes_remaining']==0 and out['registry_cleanup']['termination_needed'] is False,'HELD_AND_TREE_CLEANUP')
    need(post==out['post'] and post['same_host'] is True and post['same_boot'] is True and post['controller_device_FDs']==[] and post['owned_descendants_remaining']==0 and post['owned_device_FD_holders']=={},'ORIGINAL_POST_CLEAN')
    need(out['resources']['soft_limit_bytes']==8*1024**3 and out['resources']['soft_limit_exceeded'] is False and out['resources']['hard_limit'] is False and out['resources']['samples']>0 and out['RSS_unknown_not_zero'] is True,'ACTUAL_SAMPLED_RSS8G_NOT_HARD')
    need(intent['CPU_affinity'] and intent['CPU_affinity']==plan['fixed_CPU_affinity'],'ACTUAL_CPU2')
    need(native and native['passed'] is True and native['epoch']==a.epoch and native['deadline']==a.epoch+220 and native['denied_attempts']==[] and native['actual_updates']==0 and native['model_constructed'] is False and native['actual_model_freeze_or_trainable_initialization_qualified'] is False,'CPU_ONLY_NATIVE')
    need(native['dev_gold_read'] is False and native['blind_gold_read'] is False and native['blind_inputs_and_media_read'] is True and native['train_labels_read'] is True and native['raw_processed_payload_saved'] is False,'TRAIN_ONLY_SUPERVISION')
    need(native['framework_files']==404 and native['metadata_only_files']==13 and native['Swift_sources_verified']==7 and native['extra_wrapper_sources_verified']==3 and native['actual_versions']==plan['versions'] and native['before_device_FDs']==native['after_device_FDs']==[],'QUALIFIED_CODE_PROVIDER_SCOPE')
    need(native['native_raw_template']==native['actual_resolved_template']=='internvl_hf' and native['original_native_spec']==plan['cases']['T03.sft']['native_spec'],'ORIGINAL_NATIVE_RECIPE')
    need(progress['failed'] is False and all(native['planned_'+s]==native['completed_'+s]==result['completed'][s]==plan['planned'][s] for s in plan['planned']),'ALL144_COMPLETED')
    data=obj('data/data-binding-private.json');rows=obj('run/native/native-identities-private.json');provider=obj('run/native/processor-provider-private.json');api=obj('run/native/native-api-sources-private.json')
    need(data['planned']==plan['planned'] and data['media']==plan['media'] and data['source_inputs']==plan['source_inputs'] and data['dataset_audit_SHA256']==plan['dataset_audit']['SHA256'] and data['dataset_manifest_SHA256']==plan['dataset_manifest']['SHA256'] and data['dev_gold_read'] is False and data['blind_gold_read'] is False and data['staged_payload_files']==28 and data['staged_payload_bytes']==1031596 and data['omitted_gold_required_on_remote'] is False and data['input_only_manifest_SHA256']==digest(source['INPUT_ONLY_MANIFEST.json']),'FROZEN_DATA_BINDING')
    # Native source API identity is passive file evidence, not a new runtime invocation.
    need(set(api)=={'get_model_processor','get_template','processor_class','template_class','collator'},'EXACT_API_SOURCE_ROLES')
    allowed_roots=[Path('/home/whj/nvidia-baseline/env/lib/python3.12/site-packages').resolve(),Path('/home/whj/nvidia-baseline/env-vlm/lib/python3.12/site-packages').resolve(),Path('/usr/local/lib/python3.12/dist-packages').resolve()]
    for role,rec in api.items():
        path=Path(rec['path']);need(path.is_absolute() and any(path.resolve().is_relative_to(root) for root in allowed_roots),'QUALIFIED_PROVIDER_ROOT')
        load(path,'provider/'+role,2*1024**2,rec)
    need(len(rows)==144 and pins['run/native/native-identities-private.json']['SHA256']==native['identities_SHA256'],'ALL144_IDENTITIES_ORIGINAL_PIN')
    seen=set();tokens=0
    for i,row in enumerate(rows):
        split,ordinal=[(s,j) for s,n in plan['planned'].items() for j in range(n)][i];b=data['row_bindings'][split][ordinal]
        need(row['split']==split and row['ordinal']==ordinal and row['source_id_SHA256']==data['ordered_source_id_SHA256'][split][ordinal]==b['source_id_SHA256'] and row['source_id_SHA256'] not in seen,'ALL144_ORDERED_IDENTITIES');seen.add(row['source_id_SHA256'])
        need(row['media_SHA256']==b['media_SHA256']==plan['media'][b['media_name']]['SHA256'] and row['adapter_row_SHA256']==b['adapter_row_SHA256'] and row['full_input_retained'] is True and type(row['full_length']) is int and 0<row['full_length']<=4096,'NATIVE_INPUT_MEDIA_BINDING')
        for fs in (row['encoded_fields'],row['collated_fields']):need('input_ids' in fs and 'pixel_values' in fs and all(v['finite'] is True for v in fs.values()),'FINITE_FIELDS')
        need(row['native_EOS_ids'] and all(type(x) is int for x in row['native_EOS_ids']),'EOS_IDENTITIES')
        if split=='train':
            need(row['labels_None'] is False and row['first_supervised_index']>0 and row['supervised_EOS_count']>0 and type(row['effective_supervised_tokens']) is int and row['effective_supervised_tokens']>0,'ACTUAL_TRAIN_MASK_EOS_TOKENS');tokens+=row['effective_supervised_tokens']
        else:need(row['labels_None'] is True and row['first_supervised_index'] is None and row['supervised_EOS_count'] is None and row['effective_supervised_tokens'] is None,'DEV_BLIND_LABELS_NONE')
    need(tokens==native['effective_supervised_tokens'],'SUPERVISED_TOKEN_SUM')
    policy=json.loads(source['WRAPPER_SOURCES.json'])
    need(provider['binding_kind']=='EXACT_DECORATED_NATIVE_ENCODE_CHAIN' and provider['closure_targets_match_wrapped'] is True and provider['invocation_preserves_decorators'] is True and provider['unwrapped_leaf_called'] is False and len(provider['wrapper_chain'])==3 and provider['retry_count']==3,'REAL_DECORATED_CHAIN')
    for actual,expected in zip(provider['wrapper_chain'],policy['chain']):
        need(all(Path(actual[k]).resolve()==Path(expected['path']).resolve() for k in ('path','code_filename','inspect_sourcefile')) and actual['bytes']==expected['bytes'] and actual['file_SHA256']==expected['SHA256'] and actual['code_qualname']==expected['code_qualname'] and actual['role']==expected['role'],'WRAPPER_PROVIDER_ALIAS_RESOLVED')
    result.update(metadata_verified=True,native_preprocessing_qualified=True,state='NEW_T03_TRAIN72_DEV36_BLIND36_NATIVE_CPU_PASS',effective_supervised_tokens=tokens,finite_fields=144,labels_None=72,train_supervised_EOS=72,unique_media=24,groups=7,framework_files=404,CPU_wall_seconds=native['CPU_wall_seconds'],sampled_group_RSS_peak_bytes=out['observed_group_RSS_peak_bytes'],RSS_hard_limit=False,source_API_observations=api)
    return finish(result,pins,total)
def finish(result,pins,total):
    result.update(original_metadata_files=pins,actual_read_bytes=total,actual_metadata_files=len(pins));return result
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('request-id','publication','source-manifest-sha256','claim-sha256'):p.add_argument('--'+k,required=True)
    for k in ('scope','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('epoch','reader-origin'):p.add_argument('--'+k,type=float,required=True)
    a=p.parse_args();need(not a.output.exists(),'NEW_PRIVATE_READER_OUTPUT')
    try:r=read(a)
    except BaseException as e:r=dict(metadata_verified=False,native_preprocessing_qualified=False,planned=dict(train=72,dev=36,blind=36),error_type=type(e).__name__,error_message_SHA256=digest(str(e).encode()),process_terminal='UNKNOWN',GPU_GO=False,training_GO=False,Model_GO=False)
    a.output.write_bytes(canonical(r));print(json.dumps(r));raise SystemExit(0 if r['metadata_verified'] else 1)
