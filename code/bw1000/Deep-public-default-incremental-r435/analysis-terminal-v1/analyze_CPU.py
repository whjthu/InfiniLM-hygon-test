"""Stdlib original CPU881 receipt consumer; no producer PASS retroactivity."""
import argparse,hashlib,json
from pathlib import Path
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def analyze(observation,manifest):
 rows=observation['accounting'];assert len(rows)==1 and rows[0][0]=='910407.881'and rows[0][2:4]==['COMPLETED','0:0']
 values=[v for n,v in observation['files'].items()if n.endswith('/result.json')];assert len(values)==1
 ref=values[0];raw=ref['text'].encode();assert len(raw)==ref['bytes']and sha(raw)==ref['SHA256'];q=json.loads(raw)
 assert q['completed']and q['passed']and not q.get('error')and not q.get('traceback')and q['actual_CPU']=='910407.881'
 assert q['source_manifest_SHA256']=='29bfc3858028deb9ee7690f2f9600847d4bdd940d085cb11bc28efe4f759b840'
 assert q['GPU']is False and q['Native_dlopen']is False and q['Model_READY']is False and not q['weight_open_attempts']and not q['new_compile']and not q['old_nine_cases_replayed']
 assert q['private_default_effect']and q['Native_owner_is_recording_leaf']and not q['real_HTTP_or_Model']and not q['new_default_device_qualified']
 cases=q['cases'];assert len(cases)==10 and all(v['passed']for v in cases)
 expected=[('default_nonstream_original_body','P0','JSON',8489,None),('default_stream_true_original_body','P1','SSE',8490,True),('default_stream_missing','P0','JSON',8489,None),('explicit_P0_overrides_stream_true','P0','SSE',8490,False),('explicit_P1_overrides_stream_false','P1','JSON',8489,None)]
 hashes={8489:'681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df',8490:'d956ecfc84f1c221a7768772b868a237d02fb1e6df2b682291b9a58c9ce2e287'}
 for row,(name,mode,route,tokens,early)in zip(cases,expected):
  assert row['case']==name and row['selected']==mode and row['actual_public_route']==route and row['actual_prompt_tokens']==tokens and row['prompt_IDs_SHA256']==hashes[tokens]
  assert row['first_semantic_before_owned_finish']is early and row['complete_codec_parser_semantics_match']and not row['HTTP_or_Native_device']
 for i,kind in [(5,'timeout'),(6,'disconnect')]:
  c=cases[i];assert c['case']=='default_P1_actual_route_'+kind and c['mode']=='P1'and c['intent']==kind and c['not_finished_until_recording_owner']and c['owner_finish_preserves_abort_reason']
 assert [c['input_type']for c in cases[7:]]==['NoneType','str','int']
 assert all(c['case']=='invalid_explicit_pre_admission'and c['status']==400 and c['no_request_or_owner_work']for c in cases[7:])
 assert len(q['stages'])==4
 modules={}
 for stage in q['stages']:
  assert not stage['project_maps']and stage['torch_cuda_initialized']in[None,False]and set(stage['hidden'].values())=={'-1'}
  for name,record in stage['actual_new_modules'].items():
   prefix='/refactor/incoming/Deep-public-default-incremental-r435/cpu-v3/';assert record['path'].startswith(prefix)
   relative=record['path'][len(prefix):];assert record['SHA256']==manifest['files_SHA256'][relative];modules[name]=record
 assert {'public_deep_default_text.cli','public_deep_default_text.selector','public_deep_default_text.prepared_factory','public_deep_default_text.incremental_text'}<=set(modules)
 return dict(passed=True,actual_CPU=rows[0],raw_receipt=dict(bytes=len(raw),SHA256=ref['SHA256']),source_manifest_SHA256=q['source_manifest_SHA256'],actual_ten_cases=cases,actual_new_modules=modules,stages=4,all_project_maps_empty=True,all_hidden_minus1=True,framework_import_observed=True,torch_cuda_initialized=False,wall_s=q['wall_s'],Native_dlopen=False,GPU=False,Model=False,weights=False,compile=False,new_default_CPU_scope_qualified=True,new_default_device_qualified=False,original_failures_preserved=['910407.875 FAILED1/11s','910407.877 FAILED1/0s'],qualification='Isolated ordinary public provider source/schema + actual route/default/Deep codec/parser CPU; Native/HTTP transport remain recording leaves. Original430 explicit P1 device evidence is separate. No new Model/latency/SLO claim.')

def main():
 p=argparse.ArgumentParser();p.add_argument('observation',type=Path);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();raw=a.manifest.read_bytes();assert sha(raw)=='29bfc3858028deb9ee7690f2f9600847d4bdd940d085cb11bc28efe4f759b840'
 q=analyze(json.loads(a.observation.read_bytes()),json.loads(raw));q['observation_SHA256']=sha(a.observation.read_bytes());text=json.dumps(q,indent=2)+'\n'
 if a.output:a.output.write_text(text,encoding='utf-8')
 print(text,end='')
if __name__=='__main__':main()
