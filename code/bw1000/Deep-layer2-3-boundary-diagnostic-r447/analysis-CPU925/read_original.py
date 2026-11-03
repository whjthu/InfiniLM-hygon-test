"""Stdlib audit of the changed-entry CPU; no subprocess or project import."""
from pathlib import Path
import argparse,hashlib,json
H=lambda raw:hashlib.sha256(raw).hexdigest()
def audit(observation,previous=None):
 obs=json.loads(Path(observation).read_bytes());files={}
 for item in obs['files']:
  raw=item['text'].encode();assert len(raw)==item['bytes']and H(raw)==item['SHA256']
  files[Path(item['path']).name]=(raw,item)
 assert obs['accounting']==[['910407.925','Deep-r447-changed-entry-CPU-v2','COMPLETED','0:0','00:00:01','a14r2n20']]
 raw,item=files['result.json'];assert H(raw)=='973bb6322c64c8fcd0109ef2a49757dbf1584c493245e1f4d0c7069a8da447b5'
 q=json.loads(raw);assert q['completed']and q['passed']and q['actual_CPU']=='910407.925'
 assert not any(q[k]for k in ['Native_dlopen','Runtime_created','GPU','weights_payload_opened','actual_input_encode_replayed','old_suites_replayed'])
 assert len(q['stages'])==2 and all(not s['project_maps']and set(s['visibility'].values())=={'-1'}for s in q['stages'])
 before=q['reused_Native_derivation_CPU924'];assert before['SHA256']=='188c6cc2e7f5ca3b5ac540d97819f6d4562c18fdf5ec9be8edaafb701693cc72'and before['original_overall_passed']is False
 old=q['inherited_actual_CPU913_input'];assert old['SHA256']=='ed9728d88c06f1ef4ce7fac887c706c9caf4e4265ee3e1b713b4ef87c86dbd1c'
 assert old['actual_input']['prompt_count']==8489 and old['actual_input']['prompt_IDs_SHA256']=='681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df'
 typed=q['actual_typed_entry'];assert typed['symbols']==33 and typed['new_CABI']==7 and typed['actual_DSO_nm_checked']and typed['high64_ctx']=='0x123456789abcdef0'and typed['high64_CPUbyte_pointers']
 assert typed['actual_runner_file']=='/refactor/incoming/Deep-layer2-3-boundary-diagnostic-r447/cpu-v2/run_teacher.py'
 assert typed['actual_capture_file']=='/refactor/incoming/Deep-layer2-3-boundary-diagnostic-r447/cpu-v2/boundary_api.py'
 assert q['actual_CLI']['returncode']==0 and json.loads(q['actual_CLI']['stdout'])==dict(source_schema=True,CPU_READY=True,Native_dlopen=False,Model_READY=False,GPU=False)
 assert q['source_manifest_SHA256']=='ef23893b335815aa66ec8582b29c0b79a2a3bf0a2f05f4c2196252f1a2763311'
 cfg=json.loads(files['bound-config.json'][0]);assert cfg['modes']==[0,0]and cfg['max_forwards']==184 and cfg['raw_limit']==33554432 and not cfg['execute_authorized']
 pin=json.loads(files['BINDING_PIN.json'][0]);assert pin['actual_CPU']=='910407.921'and pin['reused_parent_objects']==48 and pin['new_HOST_TUs']==1
 previous=Path(previous)if previous else Path(observation).parent/'Deep447-entry-CPU-v1-original-observed-local.json'
 prior=json.loads(previous.read_bytes())
 for f in prior['files']:
  data=f['text'].encode();assert len(data)==f['bytes']and H(data)==f['SHA256']
  if f['path'].endswith('/CONTEXT_DERIVATION.json'):files['CONTEXT_DERIVATION.json']=(data,f)
  if f['path'].endswith('/result.json'):
   old_result=json.loads(data);assert H(data)==before['SHA256']and old_result['completed']and not old_result['passed']and old_result['actual_Native_derivation']
 proof=json.loads(files['CONTEXT_DERIVATION.json'][0]);assert proof==q['context_derivation']and proof['Native_SHA256']==pin['library_SHA256']=='8b1d1dacee6ba4b4f4d08dd38aa46c7bc529e5d3d89dc1ab753c204fe1b8084b'
 assert files['CONTEXT_DERIVATION.json'][1]['SHA256']==q['context_proof']['SHA256']=='f510393b69f198c035048f600637317419a6ef9f06947b58307e8c9b369030f0'
 source=Path(__file__).resolve().parents[1]/'entry-v2';m=json.loads((source/'SOURCE_MANIFEST.json').read_bytes());assert H((source/'SOURCE_MANIFEST.json').read_bytes())==q['source_manifest_SHA256']
 for name,h in m['files_SHA256'].items():assert H((source/name).read_bytes())==h
 for name,h in json.loads((source/'REUSE.json').read_bytes())['production_members_unchanged_from_entry_v1'].items():assert H((source/name).read_bytes())==h==H((source.parent/'entry-v1'/name).read_bytes())
 return dict(passed=True,scope='Changed7typed CABI/high64/actual nonexecute CLI only; CPU913 real encoding and CPU924 successful derivation/staging reused, no Native/Model/GPU execution',actual_CPU='910407.925 COMPLETED0/1s',producer_seconds=q['wall_s'],original_receipt_SHA256=H(raw),original_observation_SHA256=H(Path(observation).read_bytes()),source_manifest_SHA256=q['source_manifest_SHA256'],source_members=len(m['files_SHA256']),original924_FAILED_preserved=True,Native_dlopen=False,GPU=False,compile=0,actual_input_encode_replayed=False,old_suites_replayed=False,new_typed_CABI=7,high64_libffi_passed=True,actual_CLI_passed=True,Native_SHA256=pin['library_SHA256'],modes=[0,0],max_forwards=184,raw_limit=33554432,original1374_quality_FAIL_unchanged=True,root_cause='UNKNOWN',device_qualification=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('observation');p.add_argument('--previous');p.add_argument('--output');a=p.parse_args();q=audit(a.observation,a.previous);raw=json.dumps(q,indent=2)+'\n'
 if a.output:Path(a.output).write_text(raw,encoding='utf-8')
 print(raw,end='')
if __name__=='__main__':main()
