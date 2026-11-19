"""Independent new-process metadata consumer plus one fresh exact-step sacct; no payload reread."""
import argparse,hashlib,json,re,time
from pathlib import Path
D=Path(__file__).resolve().parent

def need(v,w):
    if not v:raise ValueError(w)

def stable_json(path,cap=262144):
    need(path.is_file() and not path.is_symlink(),'REGULAR_NEW_METADATA');a=path.stat();need(a.st_size<=cap,'RECEIPT_BOUND');raw=path.read_bytes();b=path.stat();need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns),'RECEIPT_STABLE')
    return json.loads(raw),dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())

def audit(run,context):
    plan=json.loads((D/'PLAN.json').read_bytes());claim=json.loads((D/'CLAIM.json').read_bytes());need(claim['READ_GO'] is True,'ROOT_ADOPTED_METADATA_READER')
    mf=(D/'SOURCE_FILES.json').read_bytes();need(hashlib.sha256(mf).hexdigest()==context['source_manifest_SHA256'],'PUBLISHED_READER_SOURCE')
    source_pins=json.loads(mf)['files']
    for name in ['read_result.py','fresh_accounting.py','delta_codec.py','common.py','PLAN.json','CLAIM.json']:
        raw=(D/name).read_bytes();need(len(raw)==source_pins[name]['bytes'] and hashlib.sha256(raw).hexdigest()==source_pins[name]['SHA256'],'EXACT_INDEPENDENT_CONSUMER_BODY')
    result=dict(independent_metadata_verified=False,planned_source_files=8,planned_output_files=7,patch_wire_cap_bytes=29360128,pins={},errors=[],asset_body_reads=0,old_receipt_reads=0,rank00_RNG_inherited=False,all26_assets_qualified=False,GPU_GO=False,Model_GO=False,training_GO=False,replay_allowed=False)
    try:
        values={}
        for name in ['intent.json','source-attempt.json','receiver-attempt.json','source-ready.json','receiver-ready.json','release.json','patch-plan.json','source-terminal.json','target-terminal.json','outcome.json']:
            value,pin=stable_json(run/name);values[name]=value;result['pins'][name]=pin
            need(all(value[k]==v for k,v in context.items()),'ONE_ORIGINAL_CONTEXT_'+name)
        local=values['outcome.json'];src=values['source-terminal.json'];host=values['target-terminal.json'];target=host['endpoint']
        from delta_codec import validate_plan
        patch=validate_plan(values['patch-plan.json']['patch_plan']);frames=patch['frames'];wire=patch['total_wire_bytes']
        need(local['patch_plan']==host['patch_plan']==patch,'SOURCE_PLAN_RELAY_HOST_EQUAL')
        need(local['passed'] is True and src['passed'] is True and host['passed'] is True and target['passed'] is True,'ALL_ORIGINAL_NORMAL_PASS')
        need(local['finished_epoch']<=context['deadline']==context['epoch']+900 and src['finished_epoch']<=context['epoch']+800 and target['finished_epoch']<=target['payload_deadline'],'SAME_ORIGINAL_DEADLINES')
        need(values['release.json']['both_ready_before_source_payload'] is True,'ROOT_HELD_RELEASE')
        control_size=3+src['control_bytes']['patch_plan_bytes']
        need(src['control_bytes']['GO_read_bytes']==target['control_bytes']['GO_read_bytes']==3 and src['control_bytes']['patch_plan_bytes']==target['control_bytes']['patch_plan_bytes'],'EXACT_RAW_GO_PLAN_FRAMING')
        need(local['io']['source_control_write_bytes']==3 and local['io']['receiver_control_write_bytes']==host['pump']['control_read_bytes']==host['pump']['control_write_bytes']==control_size,'CONTROL_BYTES_SEPARATE_FROM_PATCH_PAYLOAD')
        need(local['io']['bootstrap_write_bytes']==sum(values[m+'-attempt.json']['script_bytes']+8 for m in ['source','receiver']),'ALL_SMALL_BOOTSTRAP_BYTES')
        need(all(values[m+'-attempt.json']['framing_header_bytes']==8 and values[m+'-attempt.json']['Windows_commandline_chars']<30000 and values[m+'-attempt.json']['reference_bootstrap_base64_bytes']==0 for m in ['source','receiver']),'SHORT_C_FRAMED_SOURCE_NO_REFERENCE_PAYLOAD')
        need(local['logical_forwarded_bytes']==local['io']['logical_source_read_bytes']==local['io']['logical_target_write_bytes']==wire,'RELAY_TOTAL')
        need(set(local['transport'])=={'source','receiver'} and all(v['returncode']==0 and v['state']=='RETURNED' and v['forced_local_SSH_kill'] is False for v in local['transport'].values()),'NORMAL_HELD_SSH_TERMINALS')
        need(not local['stream_errors'] and local['failure'] is None and host['failure'] is None and src['error'] is None and target['error'] is None,'FAILURE_NOT_HIDDEN')
        need(src['owner']==values['source-ready.json']['owner'] and target['owner']==values['receiver-ready.json']['owner'],'ACTUAL_READY_OWNERS')
        need(src['owner']['UID']==1001 and target['owner']['UID']==154459 and target['owner']['host']==plan['target_node'],'SOURCE_AND_COMPUTE_TARGET')
        for x in [src,target]:
            need(x['planned_source_files']==x['source_verified_count']==8 and x['planned_output_files']==7 and x['logical_payload_bytes']==x['expected_payload_bytes']==wire,'SOURCE8_AND_WIRE_DENOMINATORS')
            need(x['source_verified_members']==patch['source_verified_members'] and x['verified_frames']==frames and x['base_unchanged_through_scope'] is True,'EIGHT_PINS_SEVEN_FRAMES_BASE_UNCHANGED')
            need(x['AS_limit_bytes']==4294967296 and x['AS_is_not_RSS'] is True and not set(x['CPU_affinity'])&{0,1},'RESOURCE_SCOPE')
            need(all(x[k]==0 for k in ['old14_accesses','old_partial_accesses','hardlinks','tensor_decodes','Torch_SDK_GPU_calls']),'NO_OTHER_ASSET_OR_MODEL_ACCESS')
            need(x['rank00_RNG_inherited'] is False and x['raw_fallback'] is False and x['post_write_initial_body_rehashed'] is False,'ORIGINAL_BYTES_NO_FAKE_RNG_NO_REREAD')
        need(len(src['CPU_affinity'])==2 and len(target['CPU_affinity'])==2 and target['allocated_CPU']==4 and target['worker_CPU']==2 and values['receiver-ready.json']['allocated_CPU']==4 and values['receiver-ready.json']['worker_CPU']==2,'ALLOCATED4_WORKER2_CPU_COUNTS')
        need(src['counters']['source_disk_read_bytes']==src['counters']['source_whole_hash_bytes']==plan['source_full_read_bytes']==1398575192,'SOURCE_READ_ALL_ORIGINAL8_EXACTLY_ONCE')
        need(src['counters']['payload_pipe_write_bytes']==src['counters']['frame_hash_bytes']==wire and src['output_completed_count']==0,'SOURCE_FRAMES_ONLY_NO_FULL_FILE_TRANSMIT')
        need(target['counters']['target_base_read_bytes']==target['counters']['target_base_hash_bytes']==plan['target_base_read_bytes']==174821899,'TARGET_BASE_READ_ONCE')
        need(target['counters']['payload_pipe_read_bytes']==target['counters']['frame_hash_bytes']==wire,'TARGET_ALL_PATCH_BYTES')
        need(target['counters']['payload_file_write_bytes']==target['counters']['target_output_hash_bytes']==plan['target_output_bytes']==1223753293 and target['output_completed_count']==7,'TARGET_ALL_SEVEN_NEW_WHOLE_FILES')
        need(src['counters']['changed_block_data_bytes']==target['counters']['changed_block_data_bytes']==src['counters']['verified_changed_block_hash_bytes']==target['counters']['verified_changed_block_hash_bytes']==sum(f['changed_data_bytes'] for f in frames),'ALL_CHANGED_BLOCK_DATA_BYTES')
        need(src['counters']['compared_original_bytes']==target['counters']['memory_base_copy_bytes']==1223753293,'LOGICAL_COMPARE_AND_COPY_BYTES')
        need(host['pump']['read_bytes']==host['pump']['write_bytes']==wire and host['pump']['completed'] is True and host['pump']['EOF'] is True and host['pump']['error'] is None and host['login_payload_file_bytes']==0,'LOGIN_RAM_FORWARD_ONLY')
        need(src['verified_frames']==target['verified_frames']==local['relay_verified_frames']==frames,'SOURCE_RELAY_TARGET_FRAME_SHA_PINS')
        for proof in [src['base_proof'],target['base_proof']]:
            need(all(proof[k]==plan['base'][k] for k in ['name','rank','bytes','SHA256']) and proof['whole_SHA_qualified'] is True and proof['stable_FD_path'] is True and proof['source_stat'][2]==174821899,'BOTH_BASE_WHOLE_SHA_STABLE_FD')
        need(len(target['verified_members'])==7,'SEVEN_NEW_RANKS_FULL_DENOMINATOR')
        inodes=set()
        for expected,row in zip(plan['files'],target['verified_members']):
            need(all(row[k]==expected[k] for k in ['name','rank','bytes','SHA256']),'EACH_OUTPUT_EQUALS_ITS_OWN_ORIGINAL_WHOLE_SHA')
            need(row['whole_SHA_qualified'] is row['file_fsync'] is row['directory_fsync'] is row['stable_FD_path'] is row['new_inode_not_base'] is True,'FSYNC_STABLE_FD_NEW_INODE')
            need(row['target_stat'][2]==174821899 and row['observer_host']==plan['target_node'],'TARGET_STABLE_NEW_INODES')
            inode=tuple(row['target_stat'][:2]);need(inode not in inodes and inode!=tuple(target['base_proof']['source_stat'][:2]),'SEVEN_DISTINCT_NEW_INODES');inodes.add(inode)
        accounting=host['accounting'];need(re.fullmatch(re.escape(plan['target_parent'])+r'\.\d+',accounting[0]) and accounting[1]==plan['label'] and accounting[2:4]==['COMPLETED','0:0'] and accounting[5]==plan['target_node'] and set(accounting[6].split(','))=={'cpu=4','mem=4G','node=1'} and target['step']==accounting[0],'ACTUAL_TERMINAL_COMPUTE_RESOURCES')
        need(host['held_srun']['reaped'] is True and host['held_srun']['termination_needed'] is False and host['held_srun']['returncode']==0,'OWNED_SRUN_CLEANUP')
        from fresh_accounting import query_once,COLS
        fresh=query_once(run,context,plan,accounting[0]);result['fresh_accounting']=fresh
        need(fresh['passed'] is True and fresh['actual_ordinary_SSH_attempts']==1 and fresh['actual_sacct_calls']==1,'ONE_FRESH_INDEPENDENT_ACTUAL_ACCOUNTING_QUERY')
        actual=fresh['record'];need(actual['columns']==COLS==host['accounting_columns'] and actual['actual_row']==accounting,'FRESH_SACCT_EQUALS_ORIGINAL_PRODUCER_TERMINAL')
        need(actual['actual_row'][0]==target['step'] and actual['actual_row'][1]==plan['label'] and actual['actual_row'][2:4]==['COMPLETED','0:0'] and actual['owner']['UID']==154459,'INDEPENDENT_FRESH_EXACT_STEP_LABEL_OWNER')
        for name in ['fresh-accounting-attempt.json','fresh-accounting-result.json']:
            _,pin=stable_json(run/name);result['pins'][name]=pin
        need(time.time()<=context['epoch']+895,'INDEPENDENT_CLOSURE_WITHIN_ORIGINAL895')
        result.update(independent_metadata_verified=True,verified_files=7,verified_original_bytes=1223753293,verified_patch_wire_bytes=wire,accounting=accounting,proof_boundary='Each own original rank wholeSHA, source8/read-once, target base/read-once, seven write/hash/fsync/stableFD, capped source/relay/receiver patch SHA, fresh exact sacct metadata closure; no independent body reread, tensor/RNG decoding, all26/model qualification',source_process_terminal='Normal exact SSH return and successful endpoint terminal; no extra PID probe',target_process_terminal='Independent fresh ordinary SSH/sacct exact step+label COMPLETED0 agrees with producer; held srun reaped')
    except (OSError,ValueError,KeyError,TypeError,IndexError) as e:result['errors'].append(dict(type=type(e).__name__,reason=str(e)[:160]))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--context',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();ctx,_=stable_json(a.context);context={k:ctx[k] for k in ['publication','request_id','epoch','deadline','source_manifest_SHA256']};r=audit(a.run,context)
    with a.output.open('x',encoding='utf8') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:r.get(k) for k in ['independent_metadata_verified','verified_files','verified_original_bytes','verified_patch_wire_bytes','errors']}));raise SystemExit(0 if r['independent_metadata_verified'] else 1)
