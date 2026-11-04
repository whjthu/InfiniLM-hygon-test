from pathlib import Path
import hashlib,json
def verify(folder):
 d=Path(folder);e=json.loads((d/'CPU_EVIDENCE.json').read_bytes());p=Path(e['raw_path'])
 assert p.is_absolute()and str(p).startswith('/public/home/aclsylqidf/hygon-inference-refactor-20261014/candidates/')
 raw=p.read_bytes();assert len(raw)==e['raw_bytes']and hashlib.sha256(raw).hexdigest()==e['raw_SHA256'];q=json.loads(raw)
 assert q['completed']is True and q['passed']is True and q['actual_CPU']=='910407.948'and q['source_manifest_SHA256']==e['source_manifest_SHA256']
 assert not q['project_maps']and all(v=='-1'for v in q['hidden'].values())and all(q[k]is False or q[k]==0 for k in ['GPU','Native_dlopen','weights','Model','compile','encoding_repeated'])
 c=q['actual_changed_metadata_consumer'];assert c['passed']and c['union_global512_original_rows']and c['firstpair_gate_ignores_gold_cap_parse']and c['firstpair_missing_actual_owner_rejected']and c['firstpair_pending_no_submission']
 assert[c['actual_eight_CLI_and_metadata'][j]['shard']for j in range(8)]==list(range(8))and all(v['actual_CLI']and v['actual_controller_and_launch_main']and v['request_count']==64 and not v['Slurm_or_Native']for v in c['actual_eight_CLI_and_metadata'])
 return dict(actual_CPU='910407.948',raw_SHA256=e['raw_SHA256'],actual_eight_metadata_passed=True,not_device_qualification=True)
