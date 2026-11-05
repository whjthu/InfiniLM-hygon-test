"""Correct private20plan metadata; original945 bytes/IDs/gold remain pinned."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json
H=lambda b:hashlib.sha256(b).hexdigest()
def prepare(source,encoding,gold,output):
 source=Path(source);m=json.loads((source/'SOURCE_MANIFEST.json').read_bytes());assert H((source/'SOURCE_MANIFEST.json').read_bytes())=='cf17247dfba2758b85e6e07251228a57d2814a1ecc56b5d7e3575e0f14ca834e'
 for n,h in m['files_SHA256'].items():assert H((source/n).read_bytes())==h
 def mapped(value):
  p=Path(value)
  if'..'in p.parts:raise ValueError('Foreign input parent traversal')
  R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
  if p.is_relative_to('/refactor'):
   return p if Path('/refactor').is_dir()else R/p.relative_to('/refactor')
  if p.is_relative_to(R):return p
  raise ValueError('Foreign private input root')
 raw=Path(encoding).read_bytes();assert H(raw)=='04c1ffa27726cc89a177814d2e8ed4838fed51052fee715a33851b9b65c8dd5a';encoded=json.loads(raw)
 raw=Path(gold).read_bytes();assert H(raw)=='c350be0128d1304ecbe5ae0d9ddd53d2d2137faac37f733b60bf0ba7cdb08a38';golds={v['id']:v for v in json.loads(raw)['cases']};records=[]
 compact=lambda q:json.dumps(q,ensure_ascii=False,separators=(',',':')).encode()
 for ordinal in [0,1,128,129]:
  meta=encoded['cases'][ordinal];raw=mapped(meta['private_path']).read_bytes();assert H(raw)==meta['private_SHA256'];v=json.loads(raw)
  assert meta['selected_ordinal']==ordinal and v['case_id']==meta['case_id']and H(compact(v['body']))==meta['request_body_SHA256']and H(compact(v['prompt_token_ids']))==meta['prompt_IDs_SHA256']
  assert meta['output_budget']==v['body']['max_tokens']==4096 and len(v['prompt_token_ids'])+4096<=32768 and v['body']['chat_template_kwargs']['enable_thinking']is False
  assert H(compact(golds[v['case_id']]))==meta['source_case_SHA256']
  records.append(dict(case_id=v['case_id'],ordinal=ordinal,body=v['body'],body_SHA256=meta['request_body_SHA256'],prompt_token_ids=v['prompt_token_ids'],prompt_IDs_SHA256=meta['prompt_IDs_SHA256'],gold=golds[v['case_id']]))
 plan=dict(schema=1,records=records,ordinals=[0,1,128,129],cases=4,HTTP=20,encoding_CPU='910407.945',new_encode=False,original_freezer_source_SHA256=m['files_SHA256']['freeze_inputs.py'])
 assert plan['ordinals']==[0,1,128,129]and len(plan['records'])==4
 # Published original had legacy descriptive labels only. Do not change original.
 plan['phase_schedule']={'A':'first4serial then16 single open wave','B':'first4serial; open4..11 cancel wave and finish transports; then12..19 STOPbacklog wave; .25s spacing within each wave'}
 plan.update(selection='Original945 fixedord0,1,128,129; original first2GSM+first2MMLU, no outcome selection',pair_order='Four serial controls then16 finite open arrivals; policyOFF/ON on independent instances',source_manifest_SHA256=H((source/'SOURCE_MANIFEST.json').read_bytes()),metadata_label_overlay=True,requests=[])
 for i in range(20):
  record=plan['records'][i%4];body=copy.deepcopy(record['body']);body['stream']=True;body['stream_options']=dict(body.get('stream_options')or{});body['stream_options']['include_usage']=True
  plan['requests'].append(dict(index=i,case_id=record['case_id'],ordinal=record['ordinal'],body=body,original_body_SHA256=record['body_SHA256'],actual_body_SHA256=H(json.dumps(body,ensure_ascii=False,separators=(',',':')).encode()),prompt_IDs_SHA256=record['prompt_IDs_SHA256'],phase='serial'if i<4 else'open',arrival_offset_s=None if i<4 else(i-4)*.25))
 with Path(output).open('x')as f:json.dump(plan,f,ensure_ascii=False);f.write('\n')
 return dict(planned=20,warm=0,fixed_ordinals=plan['ordinals'],private_plan_SHA256=H(Path(output).read_bytes()),new_encode=False)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['source','encoding','gold','output']:p.add_argument('--'+k,required=True)
 a=p.parse_args();print(json.dumps(prepare(a.source,a.encoding,a.gold,a.output)))
