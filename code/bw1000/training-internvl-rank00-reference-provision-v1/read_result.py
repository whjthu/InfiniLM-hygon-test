"""Independent original small-receipt consumer; never opens a payload or old receipt."""
import argparse,hashlib,json,re
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
    for name in ['read_result.py','fresh_accounting.py','PLAN.json','CLAIM.json']:
        raw=(D/name).read_bytes();need(len(raw)==source_pins[name]['bytes'] and hashlib.sha256(raw).hexdigest()==source_pins[name]['SHA256'],'EXACT_INDEPENDENT_CONSUMER_BODY')
    result=dict(independent_metadata_verified=False,planned_files=2,planned_payload_bytes=174847256,pins={},errors=[],asset_body_reads=0,old_receipt_reads=0,other7rank_RNG_inherited=False,all26_assets_qualified=False,GPU_GO=False,Model_GO=False,training_GO=False,replay_allowed=False)
    try:
        values={}
        for name in ['intent.json','source-ready.json','receiver-ready.json','release.json','source-terminal.json','target-terminal.json','outcome.json']:
            value,pin=stable_json(run/name);values[name]=value;result['pins'][name]=pin
            need(all(value[k]==v for k,v in context.items()),'ONE_ORIGINAL_CONTEXT_'+name)
        local=values['outcome.json'];src=values['source-terminal.json'];host=values['target-terminal.json'];target=host['endpoint']
        need(local['passed'] is True and src['passed'] is True and host['passed'] is True and target['passed'] is True,'ALL_ORIGINAL_NORMAL_PASS')
        need(local['finished_epoch']<=context['deadline']==context['epoch']+900 and src['finished_epoch']<=context['epoch']+800 and target['finished_epoch']<=target['payload_deadline'],'SAME_ORIGINAL_DEADLINES')
        need(values['release.json']['both_ready_before_source_payload'] is True,'ROOT_HELD_RELEASE')
        need(local['logical_forwarded_bytes']==local['io']['logical_source_read_bytes']==local['io']['logical_target_write_bytes']==174847256,'RELAY_TOTAL')
        need(set(local['transport'])=={'source','receiver'} and all(v['returncode']==0 and v['state']=='RETURNED' and v['forced_local_SSH_kill'] is False for v in local['transport'].values()),'NORMAL_HELD_SSH_TERMINALS')
        need(not local['stream_errors'] and local['failure'] is None and host['failure'] is None and src['error'] is None and target['error'] is None,'FAILURE_NOT_HIDDEN')
        need(src['owner']==values['source-ready.json']['owner'] and target['owner']==values['receiver-ready.json']['owner'],'ACTUAL_READY_OWNERS')
        need(src['owner']['UID']==1001 and target['owner']['UID']==154459 and target['owner']['host']==plan['target_node'],'SOURCE_AND_COMPUTE_TARGET')
        for x in [src,target]:
            need(x['planned_files']==x['completed_files']==2 and x['logical_payload_bytes']==x['expected_payload_bytes']==174847256,'TWO_WHOLE_FILE_DENOMINATOR')
            need(x['AS_limit_bytes']==4294967296 and x['AS_is_not_RSS'] is True and not set(x['CPU_affinity'])&{0,1},'RESOURCE_SCOPE')
            need(all(x[k]==0 for k in ['old14_accesses','old_partial_accesses','hardlinks','tensor_decodes','Torch_SDK_GPU_calls']),'NO_OTHER_ASSET_OR_MODEL_ACCESS')
        need(len(src['CPU_affinity'])==2 and len(target['CPU_affinity'])==2 and target['allocated_CPU']==4 and target['worker_CPU']==2 and values['receiver-ready.json']['allocated_CPU']==4 and values['receiver-ready.json']['worker_CPU']==2,'ALLOCATED4_WORKER2_CPU_COUNTS')
        need(src['counters']['source_disk_read_bytes']==174821899 and src['counters']['reference_memory_bytes']==25357 and src['counters']['payload_pipe_write_bytes']==src['counters']['hash_input_bytes']==174847256,'SOURCE_ALL_BYTES')
        need(src['counters']['reference_validation_hash_bytes']==target['counters']['reference_validation_hash_bytes']==25357 and target['counters']['reference_postread_bytes']==25357,'EXPLICIT_SMALL_REFERENCE_SECOND_READ')
        need(target['counters']['payload_pipe_read_bytes']==target['counters']['payload_file_write_bytes']==target['counters']['hash_input_bytes']==174847256,'TARGET_ALL_BYTES')
        need(host['pump']['read_bytes']==host['pump']['write_bytes']==174847256 and host['pump']['completed'] is True and host['pump']['EOF'] is True and host['pump']['error'] is None and host['login_payload_file_bytes']==0,'LOGIN_RAM_FORWARD_ONLY')
        expected=[{k:r[k] for k in ['name','bytes','SHA256']} for r in plan['files']]
        for name,rows in [('source',src['verified_members']),('receiver',target['verified_members']),('relay',local['relay_verified_members'])]:
            need([{k:r[k] for k in ['name','bytes','SHA256']} for r in rows]==expected,'INDEPENDENT_WHOLE_PINS_'+name)
        for row in src['verified_members']:
            if row['name'].startswith('initial/'):need(row['source_stat'][2]==174821899,'SOURCE_INITIAL_STABLE_FD')
        for row in target['verified_members']:need(row['target_stat'][2]==row['bytes'] and row['observer_host']==plan['target_node'],'TARGET_STABLE_NEW_INODES')
        accounting=host['accounting'];need(re.fullmatch(plan['target_parent']+r'\.\d+',accounting[0]) and accounting[1]==plan['label'] and accounting[2:4]==['COMPLETED','0:0'] and accounting[5]==plan['target_node'] and set(accounting[6].split(','))=={'cpu=4','mem=4G','node=1'} and target['step']==accounting[0],'ACTUAL_TERMINAL_COMPUTE_RESOURCES')
        need(host['held_srun']['reaped'] is True and host['held_srun']['termination_needed'] is False and host['held_srun']['returncode']==0,'OWNED_SRUN_CLEANUP')
        from fresh_accounting import query_once,COLS
        fresh=query_once(run,context,plan,accounting[0]);result['fresh_accounting']=fresh
        need(fresh['passed'] is True and fresh['actual_ordinary_SSH_attempts']==1 and fresh['actual_sacct_calls']==1,'ONE_FRESH_INDEPENDENT_ACTUAL_ACCOUNTING_QUERY')
        actual=fresh['record'];need(actual['columns']==COLS==host['accounting_columns'] and actual['actual_row']==accounting,'FRESH_SACCT_EQUALS_ORIGINAL_PRODUCER_TERMINAL')
        need(actual['actual_row'][0]==target['step'] and actual['actual_row'][1]==plan['label'] and actual['actual_row'][2:4]==['COMPLETED','0:0'] and actual['owner']['UID']==154459,'INDEPENDENT_FRESH_EXACT_STEP_LABEL_OWNER')
        for name in ['fresh-accounting-attempt.json','fresh-accounting-result.json']:
            _,pin=stable_json(run/name);result['pins'][name]=pin
        result.update(independent_metadata_verified=True,verified_files=2,verified_payload_bytes=174847256,accounting=accounting,proof_boundary='Original source/relay/compute fullSHA and write/fsync/stableFD metadata closure; no independent body reread, no tensor/RNG decoding, no all26 qualification',source_process_terminal='Normal exact SSH return and successful endpoint terminal; no extra PID probe',target_process_terminal='Independent fresh ordinary SSH/sacct exact step+label COMPLETED0 agrees with original accounting; held srun reaped')
    except (OSError,ValueError,KeyError,TypeError,IndexError) as e:result['errors'].append(dict(type=type(e).__name__,reason=str(e)[:160]))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--context',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();ctx,_=stable_json(a.context);context={k:ctx[k] for k in ['publication','request_id','epoch','deadline','source_manifest_SHA256']};r=audit(a.run,context)
    with a.output.open('x',encoding='utf8') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:r.get(k) for k in ['independent_metadata_verified','verified_files','verified_payload_bytes','errors']}));raise SystemExit(0 if r['independent_metadata_verified'] else 1)
