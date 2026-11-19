"""Hard bound to the original failed/unknown three-stream identity; no replacement context or sources."""
import hashlib,json,re,stat
from pathlib import Path
from common import P,DEP,need,canonical,sha

def stable_json(path,cap=262144):
    need(path.is_file() and not path.is_symlink(),'ORIGINAL_SMALL_REGULAR_RECEIPT_REQUIRED');before=path.stat();need(stat.S_ISREG(before.st_mode) and before.st_size<=cap,'ORIGINAL_RECEIPT_BOUND');raw=path.read_bytes();after=path.stat()
    need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'ORIGINAL_RECEIPT_STABLE')
    return json.loads(raw),dict(bytes=len(raw),SHA256=sha(raw))

def validate_snapshot(v):
    need(v['schema']=='ORIGINAL_THREE_STREAM_PASS_SNAPSHOT_V1' and v['original_context']==DEP['original_context'],'NO_ALTERNATE_ORIGINAL_CONTEXT')
    need(v['original_passed'] is v['original_independent_passed'] is True and v['verified_files']==3 and v['verified_blocks']==11590 and v['verified_raw_bytes']==P['read3_bytes'],'ACTUAL_ORIGINAL_THREE_OF_THREE_PASS_REQUIRED')
    need(len(v['members'])==3 and v['original_MF_SHA256']==DEP['original_context']['source_manifest_SHA256'],'ORIGINAL_SOURCE_IDENTITY_REQUIRED')
    for row,m in zip(P['read3'],v['members']):
        need(all(m[k]==row[k] for k in ['name','bytes','SHA256','file_id']),'EXACT_ORIGINAL_THREE_SHARD_PINS')
        need(m['normal_source_SSH'] is m['normal_target_SSH'] is True and m['source_whole_SHA256']==m['target_whole_SHA256']==row['SHA256'],'ORIGINAL_SOURCE_TARGET_NORMAL_WHOLE_SHA')
        a=m['accounting'];need(re.fullmatch(re.escape(row['original_parent'])+r'\.\d+',a[0]) and a[1]==row['original_label'] and a[2:4]==['COMPLETED','0:0'] and a[5]==row['original_node'] and set(a[6].split(','))=={'cpu=4','mem=4G','node=1'},'ORIGINAL_ALL3_C0_RESOURCES_REQUIRED')
        need(m['fresh_original_accounting'] is True and m['target_proof']['stable_FD_path'] is m['target_proof']['file_fsync'] is m['target_proof']['directory_fsync'] is True,'ORIGINAL_INDEPENDENT_FRESH_ACCOUNTING_AND_STABLE_WRITE_REQUIRED')
    need(len(canonical(v))<=32768,'BOUNDED_UPSTREAM_SNAPSHOT');return v

def collect_original_pass(G):
    # This is a prerequisite metadata gate, never an execution/reconciliation/retry of the old producer.
    root=G/DEP['original_dispatch'];pins={};expected=DEP['original_context']
    def take(name,ctx=expected):
        value,pin=stable_json(root/name);pins[name]=pin;need(all(value[k]==x for k,x in ctx.items()),'EXACT_ORIGINAL_CONTEXT_'+name);return value
    intent=take('intent.json');need(pins['intent.json']=={k:DEP['original_intent'][k] for k in ['bytes','SHA256']},'ROOT_EXACT_ORIGINAL_INTENT_PIN')
    closure=take('closure.json');need(closure['passed'] is closure['producer_passed'] is closure['independent_metadata_verified'] is True and closure['verified_files']==3 and closure['verified_raw_bytes']==P['read3_bytes'],'ORIGINAL_FAILED_UNKNOWN_MISSING_IS_NOT_PASS_NO_NEW_SSH')
    outcome=take('outcome.json');independent,ipin=stable_json(root/'independent-result.json');pins['independent-result.json']=ipin
    need(outcome['passed'] is True and outcome['completed_files']==3 and independent['independent_metadata_verified'] is True and independent['verified_files']==3 and independent['verified_blocks']==11590 and independent['verified_raw_bytes']==P['read3_bytes'],'ORIGINAL_FULL_INDEPENDENT_CLOSURE_REQUIRED')
    members=[]
    for row in P['read3']:
        fid=row['file_id'];ctx=dict(expected,file_id=fid);prefix='shard%02d/'%fid;src=take(prefix+'source-terminal.json',ctx);host=take(prefix+'target-terminal.json',ctx);target=host['endpoint'];arm=outcome['arms'][str(fid)];audit=independent['arms'][str(fid)]
        need(src['passed'] is host['passed'] is target['passed'] is arm['passed'] is audit['verified'] is True,'ORIGINAL_FILE_NOT_FULLY_VERIFIED')
        need(all(target[k]==x for k,x in ctx.items()) and src['raw_SHA256']==target['raw_SHA256']==row['SHA256'],'ORIGINAL_ENDPOINT_CONTEXT_AND_WHOLE_SHA')
        need(src['wire_SHA256']==target['wire_SHA256']==arm['wire_SHA256'] and src['wire_processed_bytes']==target['wire_processed_bytes']==arm['wire_forwarded_bytes'],'ORIGINAL_ALL_WIRE_HASH_BYTES')
        transports=[outcome['transport'][str(fid)+'-'+mode] for mode in ['source','receiver']];need(all(t['state']=='RETURNED' and t['returncode']==0 and t['forced_local_SSH_kill'] is False and t['held_SSH_reaped'] is True for t in transports),'ORIGINAL_NO_UNKNOWN_TRANSPORT')
        fresh=audit['fresh_accounting'];need(fresh['passed'] is True and fresh['actual_ordinary_SSH_attempts']==fresh['actual_sacct_calls']==1 and fresh['record']['actual_row']==host['accounting'] and target['step']==host['accounting'][0],'ORIGINAL_FRESH_ACCOUNTING_REQUIRED')
        need(target['file_proof']['name']==row['name'] and target['file_proof']['bytes']==row['bytes'] and target['file_proof']['SHA256']==row['SHA256'],'ORIGINAL_TARGET_PROOF_EXACT_PIN')
        members.append(dict(name=row['name'],bytes=row['bytes'],SHA256=row['SHA256'],file_id=fid,source_whole_SHA256=src['raw_SHA256'],target_whole_SHA256=target['raw_SHA256'],normal_source_SSH=True,normal_target_SSH=True,accounting=host['accounting'],fresh_original_accounting=True,target_proof=target['file_proof'],target_owner=target['owner']))
    return validate_snapshot(dict(schema='ORIGINAL_THREE_STREAM_PASS_SNAPSHOT_V1',original_context=expected,original_MF_SHA256=expected['source_manifest_SHA256'],original_passed=True,original_independent_passed=True,verified_files=3,verified_blocks=11590,verified_raw_bytes=P['read3_bytes'],members=members,original_receipt_pins=pins,boundary='Only the fixed original request can satisfy this gate. Current originalFAIL/UNKNOWN cannot; no alternate request, weight path, new transfer or silent upgrade.'))
