"""New-process source/relay/compute metadata consumer, plus one fresh exact sacct per known arm."""
import argparse,hashlib,json,re,time
from pathlib import Path
from common import P,CONTEXT_KEYS,need,canonical
D=Path(__file__).resolve().parent

def read(path):
    need(path.is_file() and not path.is_symlink(),'REGULAR_NEW_RECEIPT');a=path.stat();need(a.st_size<=262144,'SMALL_RECEIPT_BOUND');raw=path.read_bytes();b=path.stat();need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns,b.st_ctime_ns),'STABLE_RECEIPT')
    return json.loads(raw),dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())

def audit(run,context):
    from verify_package import verify
    checked=verify(D);need(checked['READ_GO'] is True and checked['MF_SHA256']==context['source_manifest_SHA256'],'ROOT_ADOPTED_EXACT_CONSUMER_SOURCE')
    result=dict(independent_metadata_verified=False,planned_files=3,planned_blocks=11590,planned_raw_bytes=P['raw_bytes'],verified_files=0,verified_blocks=0,verified_raw_bytes=0,arms={},pins={},errors=[],asset_body_reads=0,old_receipt_reads=0,Slurm_submissions=0,old14_accesses=0,initial362_364_accesses=0,assembly_actions=0,GPU_GO=False,Model_GO=False,training_GO=False,replay_allowed=False)
    def receipt(path,ctx):
        value,pin=read(path);need(all(value[k]==v for k,v in ctx.items()),'ONE_ORIGINAL_CONTEXT_'+path.name);result['pins'][path.relative_to(run).as_posix()]=pin;return value
    # A failed READY gate does not erase the raw controls or the host's original handle.
    from fresh_accounting import query_once,COLS
    for row in P['files']:
        fid=row['file_id'];ctx=dict(context,file_id=fid);arm=run/('shard%02d'%fid);a=dict(verified=False,planned_blocks=row['blocks'],planned_raw_bytes=row['bytes'],errors=[],fresh_accounting=None);result['arms'][str(fid)]=a
        try:
            attempt=receipt(arm/'receiver-attempt.json',ctx);need(attempt['exec_request_attempted'] is True,'ORIGINAL_RECEIVER_ATTEMPT')
            candidates=[];a['observed_control_stages']=[]
            def add_handle(value):
                need(isinstance(value,str) and re.fullmatch(re.escape(row['parent'])+r'\.\d+',value),'ORIGINAL_CONTROL_NUMERIC_HANDLE');candidates.append(value)
            def remember(value):
                need(isinstance(value,dict) and all(value.get(k)==v for k,v in ctx.items()),'ORIGINAL_CONTROL_CONTEXT')
                if value.get('mode')=='receiver':
                    owner=value.get('owner');need(isinstance(owner,dict) and owner.get('UID')==154459 and owner.get('host')==row['node'],'ORIGINAL_CONTROL_RECEIVER_OWNER')
                    add_handle(value.get('step'))
                for key in ['observed_compute_ready','endpoint']:
                    if isinstance(value.get(key),dict):remember(value[key])
                accounting=value.get('accounting')
                if accounting is not None:
                    need(isinstance(accounting,list) and len(accounting)==7 and accounting[1]==row['label'] and accounting[5]==row['node'],'ORIGINAL_CONTROL_ACCOUNTING_BINDING');add_handle(accounting[0])
            ready_path=arm/'receiver-ready.json'
            try:
                if ready_path.exists():remember(receipt(ready_path,ctx))
            except (OSError,ValueError,KeyError,TypeError,IndexError) as e:
                a['errors'].append(dict(stage='accepted_ready_auxiliary',type=type(e).__name__,reason=str(e)[:160]))
            for ordinal in (0,1):
                raw_path=arm/('receiver-control-%02d.private'%ordinal);pin_path=raw_path.with_name(raw_path.name+'.json')
                try:
                    if not raw_path.exists() and not pin_path.exists():continue
                    pin=receipt(pin_path,ctx);value,actual=read(raw_path);result['pins'][raw_path.relative_to(run).as_posix()]=actual
                    need(pin['path']==raw_path.name and pin['bytes']==actual['bytes']<=65536 and pin['SHA256']==actual['SHA256'] and pin['truncated'] is False,'ORIGINAL_CONTROL_RAW_PIN')
                    need(isinstance(value,dict),'ORIGINAL_CONTROL_OBJECT');a['observed_control_stages'].append(value.get('stage','UNKNOWN'));remember(value)
                except (OSError,ValueError,KeyError,TypeError,IndexError) as e:
                    a['errors'].append(dict(stage='preserved_original_control',type=type(e).__name__,reason=str(e)[:160]))
            need(all(isinstance(v,str) and re.fullmatch(re.escape(row['parent'])+r'\.\d+',v) for v in candidates) and len(set(candidates))<=1,'ORIGINAL_CONTROL_HANDLE_CONFLICT')
            step=candidates[0] if candidates else None
            # The same one SSH also reads exactly three new-host metadata files;
            # a missing local READY never implies that no receiver was submitted.
            fresh=query_once(arm,ctx,dict(target_parent=row['parent'],label=row['label'],node=row['node'],target_namespace=row['target_namespace']),step);a['fresh_accounting']=fresh
            a['known_original_step']=fresh.get('step');a['fresh_terminal_row']=fresh.get('record',{}).get('actual_row') if isinstance(fresh.get('record'),dict) else None
            for name in ['fresh-accounting-attempt.json','fresh-accounting-result.json']:
                path=arm/name
                if path.exists():_,pin=read(path);result['pins'][path.relative_to(run).as_posix()]=pin
        except (OSError,ValueError,KeyError,TypeError,IndexError) as e:a['errors'].append(dict(stage='fresh_exact_step',type=type(e).__name__,reason=str(e)[:160]))
    try:
        intent=receipt(run/'intent.json',context);release=receipt(run/'release.json',context);local=receipt(run/'outcome.json',context)
        need(release['all_six_READY_before_any_body_GO'] is True and release['source_masks']==[[10,11],[12,13],[14,15]],'ALL_SIX_READY_FIXED6CPUS_BEFORE_GO')
        need(local['planned_files']==3 and local['planned_blocks']==11590,'FULL_GROUP_FAILURE_DENOMINATOR')
        need(local['PC_payload_file_bytes']==0 and all(local[k]==0 for k in ['old14_accesses','initial362_364_accesses','old_partial_accesses','assembly_actions']),'NO_OTHER_ASSET_ACCESS_OR_ASSEMBLY')
        source_owners=[];markers=[];output_inodes=[]
        for row in P['files']:
            fid=row['file_id'];ctx=dict(context,file_id=fid);arm=run/('shard%02d'%fid);a=result['arms'][str(fid)]
            try:
                values={n:receipt(arm/n,ctx) for n in ['intent.json','release.json','source-attempt.json','receiver-attempt.json','source-ready.json','receiver-ready.json','source-terminal.json','target-terminal.json','relay-outcome.json']}
                src=values['source-terminal.json'];host=values['target-terminal.json'];target=host['endpoint'];relay=values['relay-outcome.json'];sr=values['source-ready.json'];tr=values['receiver-ready.json'];fresh=a['fresh_accounting']
                need(src['passed'] is host['passed'] is target['passed'] is relay['passed'] is True,'ALL_ORIGINAL_NORMAL_PASS')
                need(src['error'] is target['error'] is host['failure'] is relay['failure'] is None,'FAILURE_NOT_HIDDEN')
                need(src['owner']==sr['owner'] and target['owner']==tr['owner'] and src['owner']['UID']==1001 and target['owner']['UID']==154459 and target['owner']['host']==row['node'],'ACTUAL_ORIGINAL_READY_OWNERS')
                source_owners.append(src['owner']);markers.append(tr['common_marker']);need(all(tr['common_marker']['observer'][k]==target['owner'][k] for k in ['UID','host','boot_id','PID','startticks']),'MARKER_COMPUTE_OBSERVER_HOST_BOOT_OWNER')
                need(src['CPU_affinity']==row['source_CPU_affinity'] and len(target['CPU_affinity'])==2 and not set(target['CPU_affinity'])&{0,1},'FIXED_SOURCE_AND_TARGET_MASKS')
                need(src['allocated_CPU']==src['worker_CPU']==target['worker_CPU']==2 and target['allocated_CPU']==tr['allocated_CPU']==4 and tr['GRES']==0,'ALLOC4_WORKER2_SOURCE2_NO_GRES')
                for endpoint in [src,target]:
                    need(endpoint['deadline']==context['epoch']+86400 and endpoint['finished_epoch']<=endpoint['payload_deadline']<=context['epoch']+84600,'ORIGINAL_WORKER_DEADLINE')
                    need(endpoint['planned_files']==endpoint['completed_files']==1 and endpoint['group_files']==3 and endpoint['planned_blocks']==endpoint['blocks_completed']==row['blocks'] and endpoint['raw_processed_bytes']==row['bytes'],'ONE_FILE_WHOLE_BLOCK_DENOMINATOR')
                    need(endpoint['raw_SHA256']==row['SHA256'] and endpoint['AS_limit_bytes']==4294967296 and endpoint['AS_is_not_RSS'] is True and endpoint['nice']>=10,'ORIGINAL_WHOLE_SHA_AND_RESOURCE_LIMITS')
                    need(endpoint['zlib']['level']==1 and endpoint['zlib']['wbits']==15 and endpoint['zlib']['independent_blocks'] is True and endpoint['GPU_calls']==0,'OPAQUE_STDLIB_CPU_ZLIB')
                    need(all(endpoint[k]==0 for k in ['old14_accesses','old_partial_accesses','initial362_364_accesses','assembly_actions','hardlinks','tensor_decodes']) and endpoint['raw_fallback'] is False,'NO_OTHER_ASSET_FALLBACK_OR_MODEL')
                    need(endpoint['counters']['GO_control_read_bytes']==3,'RAW_EXACT_GO3')
                wire=src['wire_processed_bytes'];compressed=wire-row['blocks']*128
                need(0<wire<=row['max_wire_bytes'] and src['wire_SHA256']==target['wire_SHA256']==relay['wire_SHA256'] and target['wire_processed_bytes']==relay['wire_forwarded_bytes']==wire,'SOURCE_RELAY_TARGET_ROLLING_WIRE_SHA_ALL_HEADERS_BODY')
                need(src['counters']['source_disk_read_bytes']==src['counters']['source_whole_hash_bytes']==src['counters']['raw_block_hash_bytes']==row['bytes'],'SOURCE_FULL_READ_AND_WHOLE_HASH_ONCE')
                need(src['counters']['compressed_produced_bytes']==src['counters']['compressed_block_hash_bytes']==target['counters']['compressed_block_hash_bytes']==target['counters']['decompress_input_bytes']==compressed,'ALL_COMPRESSED_BYTES')
                need(target['counters']['decoded_raw_bytes']==target['counters']['raw_block_hash_bytes']==target['counters']['target_whole_hash_bytes']==target['counters']['payload_file_write_bytes']==row['bytes'],'TARGET_DECODE_WRITE_HASH_ALL_RAW_BYTES')
                need(src['counters']['wire_hash_bytes']==src['counters']['payload_pipe_write_bytes']==target['counters']['wire_hash_bytes']==target['counters']['payload_pipe_read_bytes']==wire,'SOURCE_TARGET_ALL_WIRE_BYTES')
                io=relay['io'];need(io['source_read_bytes']==io['target_write_bytes']==io['wire_hash_bytes']==wire and io['compressed_hash_bytes']==compressed and io['source_control_write_bytes']==io['receiver_control_write_bytes']==3,'PC_ALL_BYTES_SEPARATE_FROM_CONTROL')
                need(io['bootstrap_write_bytes']==sum(values[m+'-attempt.json']['script_bytes']+8 for m in ['source','receiver']) and all(values[m+'-attempt.json']['commandline_chars']<30000 for m in ['source','receiver']),'SHORT_C_AND_EXACT_SOURCE_FRAMING_BYTES')
                need(relay['raw_forwarded_bytes']==row['bytes'] and relay['blocks_forwarded']==row['blocks'],'PC_ALL_RAW_LENGTHS_BLOCKS')
                need(host['pump']['read_bytes']==host['pump']['write_bytes']==wire and host['pump']['blocks_completed']==row['blocks'] and host['pump']['control_read_bytes']==host['pump']['control_write_bytes']==3 and host['pump']['completed'] is host['pump']['EOF'] is True and host['pump']['error'] is None and host['login_payload_file_bytes']==0,'HOST_BOUNDED_RAM_ONLY_ALL_BYTES')
                for e in [src,target]:
                    proof=e['file_proof'];need(proof['name']==row['name'] and proof['bytes']==row['bytes'] and proof['SHA256']==row['SHA256'] and proof['stable_FD_path'] is True,'OWN_ORIGINAL_FULL_SHA_STABLE_FD')
                need(src['file_proof']['source_stat'][2]==row['bytes'],'SOURCE_ORIGINAL_STABLE_SIZE')
                proof=target['file_proof'];need(proof['target_stat'][2]==row['bytes'] and proof['observer_host']==row['node'] and proof['observer_boot_id']==target['owner']['boot_id'] and proof['observer_UID']==154459 and proof['file_fsync'] is proof['pre_rename_directory_fsync'] is proof['directory_fsync'] is proof['new_exclusive_inode'] is True,'OUTPUT_FILE_DIR_FSYNC_BEFORE_RENAME_AND_AFTER')
                output_inodes.append(proof['target_stat'][1]);need(target['common_marker']==tr['common_marker'],'COMMON_MARKER_UNCHANGED_THROUGH_WRITE')
                adm=tr['fresh_admission'];need(adm['parent']['lease_end_epoch']>=context['deadline'] and adm['actual_end_epoch']==adm['actual_start_epoch']+adm['actual_limit_seconds']<=context['epoch']+85020 and adm['actual_limit_seconds']==adm['requested_seconds'] and adm['requested_seconds']%60==0,'FRESH_PARENT_LEASE_ACTUAL_STEP_END')
                queries=host['step_visibility_queries'];need(1<=len(queries)<=3,'AT_MOST_THREE_SAME_HANDLE_VISIBILITY_QUERIES')
                need(all(q['requested_step']==target['step'] and q['ordinal']==i+1 and q['query_returncode']==0 and q['error_type'] is None and 0<q['query_timeout_seconds']<=5 and q['query_wall_seconds']<=15 and q['total_elapsed_seconds']<=15 and q['matched_records']==(1 if i==len(queries)-1 else 0) and not q['stdout']['truncated'] and not q['stderr']['truncated'] for i,q in enumerate(queries)),'ACTUAL_BOUNDED_SAME_HANDLE_VISIBILITY')
                need(queries[-1]['selected_fields']==adm['step_fields'],'ACTUAL_SELECTED_STEP_FIELDS_NOT_REQUESTED_FALLBACK')
                hh,mm,ss=map(int,adm['time_arg'].split(':'));need(mm<60 and ss<60 and hh*3600+mm*60+ss==adm['requested_seconds'],'ACTUAL_CORRECT_HH_MM_SS')
                accounting=host['accounting'];need(accounting[0]==target['step']==tr['step'] and accounting[1]==row['label'] and accounting[2:4]==['COMPLETED','0:0'] and accounting[5]==row['node'] and set(accounting[6].split(','))=={'cpu=4','mem=4G','node=1'},'ORIGINAL_COMPUTE_C0_RESOURCES')
                need(host['held_srun']['reaped'] is True and host['held_srun']['termination_needed'] is False and host['held_srun']['returncode']==0,'HELD_SRUN_NORMAL_REAP')
                need(fresh and fresh['passed'] is True and fresh['actual_ordinary_SSH_attempts']==fresh['actual_sacct_calls']==1 and fresh['record']['columns']==host['accounting_columns']==COLS and fresh['record']['actual_row']==accounting,'INDEPENDENT_FRESH_EXACT_SACCT_EQUALS_PRODUCER')
                need(fresh['record']['requested_step']==target['step'] and fresh['record']['requested_label']==row['label'] and fresh['record']['owner']['UID']==154459,'FRESH_EXACT_STEP_LABEL_OWNER')
                for mode in ['source','receiver']:
                    t=local['transport'][str(fid)+'-'+mode];need(t['state']=='RETURNED' and t['returncode']==0 and t['forced_local_SSH_kill'] is False and t['held_SSH_reaped'] is True,'NORMAL_HELD_SSH_NO_UNKNOWN')
                need(not a['errors'],'PRESERVED_CONTROL_ERRORS_NOT_HIDDEN')
                a.update(verified=True,verified_blocks=row['blocks'],verified_raw_bytes=row['bytes'],wire_bytes=wire,wire_SHA256=src['wire_SHA256'])
            except (OSError,ValueError,KeyError,TypeError,IndexError) as e:a['errors'].append(dict(stage='original_metadata',type=type(e).__name__,reason=str(e)[:160]))
        result['verified_files']=sum(a['verified'] for a in result['arms'].values());result['verified_blocks']=sum(a.get('verified_blocks',0) for a in result['arms'].values());result['verified_raw_bytes']=sum(a.get('verified_raw_bytes',0) for a in result['arms'].values())
        need(result['verified_files']==3 and result['verified_blocks']==11590 and result['verified_raw_bytes']==P['raw_bytes'],'ALL_THREE_FILES_FULL_FAILURE_DENOMINATOR')
        need(len({o['host'] for o in source_owners})==len({o['boot_id'] for o in source_owners})==1 and len({o['PID'] for o in source_owners})==3,'THREE_ORIGINAL_SOURCE_PROCESSES_SAME_HOST_BOOT')
        need(len({(m['bytes'],m['SHA256'],m['owner_UID']) for m in markers})==1 and all(m['owner_UID']==154459 and m['stable_FD_path'] is True and m['stat']==m['stat_after'] for m in markers),'SAME_CONTEXT_MARKER_PER_OBSERVER_STABILITY_NO_CROSS_MOUNT_IDENTITY_CLAIM')
        need(local['passed'] is True and local['completed_files']==3 and local['blocks_forwarded']==11590 and local['raw_forwarded_bytes']==P['raw_bytes'] and local['wire_forwarded_bytes']==sum(a['wire_bytes'] for a in result['arms'].values()),'PRODUCER_THREE_OF_THREE_NOT_UPGRADED')
        need(time.time()<=context['epoch']+P['budget']['consumer_closure'],'ORIGINAL_CONSUMER_CLOSURE_DEADLINE')
        result.update(independent_metadata_verified=True,proof_boundary='Opaque3 original shards only: source single-read wholeSHA, target streaming bounded decode/wholeSHA/write/fsync/stableFD,3 rolling wire SHA,6 held normalSSH,3 fresh exact actualsacct. No cross-observer st_dev/inode comparison or proof one compute observer sees all3 outputs; later assembly requires separate single-observer metadata/strongpin audit. No post-write body reread, old14/initial/17/26 assembly, model/native/training qualification.')
    except (OSError,ValueError,KeyError,TypeError,IndexError) as e:result['errors'].append(dict(type=type(e).__name__,reason=str(e)[:160]))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--context',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();intent,_=read(a.context);context={k:intent[k] for k in CONTEXT_KEYS};result=audit(a.run,context)
    with a.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ['independent_metadata_verified','planned_files','verified_files','planned_blocks','verified_blocks','verified_raw_bytes','errors']}));raise SystemExit(0 if result['independent_metadata_verified'] else 1)
