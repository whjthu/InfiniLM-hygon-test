"""Bind changed semantic clients to CPU635; reuse DSO evidence only for unchanged code."""
import json,hashlib,pathlib
OWNER_SHA='076fdb2e6bf89d0cd2adf4cc7b26dea13f47556b7efa88c936f0db3556e988f7'
PARENT_SHA='415af44d184e05b2c9b5c5971800006c6a07eb2485bc3f08c5943574b8b3cde4'
CPU635_SHA='5ae41a5de00f39005e6e0ee04ea1ce52c5c9e2a114fd6c9924394b6d0c2ab57c'
EXPECTED_CASES={'positive-bare','positive-fence','fence-prefix','fence-suffix','duplicate-final-key','nonfinite-final','stale-transport','wrong-manifest','wrong-fixture','wrong-profile-pin','wrong-owner-index','wrong-final-nonce','quality-owner-index','quality-cap','quality-clock'}
CHANGED={'client_multicall_r281.py','client_quality_r281.py','GPU_inside_r277.sh','worker_r206b.py','PREPARATION.json'}
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def validate_receipt(c):
 assert c['completed'] is True and c['passed'] is True and c['CPU_only'] is True
 assert c['actual_step']=='910407.635' and c['owner_source_manifest_SHA256']==OWNER_SHA
 assert all(type(c[k]) is int and c[k]==0 for k in ['GPU','Model','production_HTTP'])
 assert c['synthetic_owner_and_quality_only'] is True and c['real_multicall_main_quality_main_transport_and_validators'] is True
 assert c['actual_preweight_and_legacy_dependency_import_guarded_PASS'] is True and c['resolved_metadata_correct'] is True
 assert len(c['cases'])==15 and {x['name'] for x in c['cases']}==EXPECTED_CASES and all(x['passed'] is True for x in c['cases'])
 for x in c['cases']:
  if x['name'].startswith('positive-'):assert x['real_multicall_returncode']==0 and x['quality_returncode']==0 and x['synthetic_HTTP']==68
 assert c['loader_AST']['Native_forward_AST_unchanged'] and c['loader_AST']['original_finally_destroy_retained']
 return c
def verify_supplement(root,source):
 root=pathlib.Path(root);source=pathlib.Path(source);base=root/'incoming/Deep-latest-full-owner-r300'
 assert sha(base/'SOURCE_MANIFEST.json')==PARENT_SHA and sha(source/'SOURCE_MANIFEST.json')==OWNER_SHA
 a=json.loads((base/'SOURCE_MANIFEST.json').read_bytes());b=json.loads((source/'SOURCE_MANIFEST.json').read_bytes())
 assert b['original_parent_manifest_SHA256']==PARENT_SHA
 assert set(b['files_SHA256'])==set(a['files_SHA256'])|{'content_json.py'}
 assert {n for n,h in a['files_SHA256'].items() if b['files_SHA256'][n]!=h}==CHANGED
 for directory,m in [(base,a),(source,b)]:
  assert all(sha(directory/n)==h for n,h in m['files_SHA256'].items())
 assert (source/'GPU_inside_r277.sh').read_text()==(base/'GPU_inside_r277.sh').read_text().replace('Deep-latest-full-owner-r300','Deep-semantic68-owner-r320')
 p=json.loads((source/'PREPARATION.json').read_bytes());assert p['HTTP']==p['Native']==68
 cpath=root/'candidates/Deep-semantic68-CPU-r321/CPU-prepared.json';assert sha(cpath)==CPU635_SHA
 validate_receipt(json.loads(cpath.read_bytes()))
 return {'owner_manifest_SHA256':OWNER_SHA,'CPU635_receipt_SHA256':CPU635_SHA,'actual_CPU':'910407.635','whole_client_CPU_cases':15,'unchanged_parent_DSO_loader_cleanup_reused':True,'real_quality_or_performance_proved':False}
