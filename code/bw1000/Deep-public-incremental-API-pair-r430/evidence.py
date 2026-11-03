"""Consume both original CPU scopes; never trust863's premature passed flag."""
from pathlib import Path
import hashlib,importlib.util,json
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def check(deploy,package,*,local_files=None):
 card=json.loads((Path(deploy)/'CPU_EVIDENCE.json').read_bytes());observations={};paths={}
 for ref in card['references']:
  path=Path(ref['path'])if local_files is None else Path(local_files)/ref['local_untracked_file']
  raw=path.read_bytes();assert sha(raw)==ref['SHA256'];paths[ref['kind']]=path;observations[ref['kind']]=json.loads(raw)
 source=Path(package)/'cpu-v3';assert sha((source/'SOURCE_MANIFEST.json').read_bytes())==card['metadata_CPU_source_MF']
 for name,h in json.loads((source/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha((source/name).read_bytes())==h,name
 spec=importlib.util.spec_from_file_location('Deep430_original_functional_closure',source/'verify_existing_cases.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 functional=module.verify(paths['functional9_failed_producer']);assert functional['functional_case_scope_passed']and not functional['retroactive_producer_PASS']
 archive=observations['metadata_completed0'];assert len(archive['accounting'])==1 and archive['accounting'][0][0]=='910407.866'and archive['accounting'][0][2:4]==['COMPLETED','0:0']
 records=list(archive['files'].values());assert len(records)==1;r=records[0];raw=r['text'].encode();assert len(raw)==r['bytes']and sha(raw)==r['SHA256']==card['metadata_original_raw_SHA256']
 metadata=json.loads(raw);assert metadata['completed']and metadata['passed']and not metadata.get('error')and not metadata.get('cleanup_error')
 assert metadata['source_manifest_SHA256']==card['metadata_CPU_source_MF']and metadata['production_source_manifest_SHA256']==card['production_source_MF']and metadata['CPU_v2_source_manifest_SHA256']==card['functional_CPU_source_MF']
 assert metadata['Native_dlopen']is False and not metadata['nine_functional_cases_replayed']and metadata['Native_owner_calls']==metadata['decoder_prefix_or_parser_case_replays']==0 and not metadata['weight_open_attempts']
 assert len(metadata['stages'])==5
 for stage in metadata['stages']:assert not stage['all_project_map_matches']and set(stage['hidden_GPU_env'].values())=={'-1'}and stage['torch_cuda_Python_initialized']in [None,False]
 return dict(functional9_scope=True,original_CPU863_state='FAILED1/error preserved',metadata_actual_CPU='910407.866 COMPLETED0',metadata_maps_empty=True,metadata_framework_import_observed=True,both_distinct_evidence_scopes_consumed=True,Native_or_device_qualification=False)
