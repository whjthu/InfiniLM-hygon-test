"""Explicit CPU mock admission regressions; never a Native/GPU result."""
from pathlib import Path
import json,copy,hashlib
from pilot_data_contract_r168a import validate_payloads,read_pinned,IDS,GOLD_REMOTE_SHA

def run_cases(root):
 root=Path(root);root.mkdir(exist_ok=False,parents=True);model=root/'mock_model';model.mkdir();src=Path(__file__).resolve().parent
 gold=json.loads((src/'dataset64.json').read_bytes())['cases'];other=next(x['id']for x in gold if x['id']not in IDS)
 cfg={'eos_token_id':1,'num_hidden_layers':43,'num_nextn_predict_layers':1,'compress_ratios':[0,0]+[128]*20+[4]*21+[128],'vocab_size':1000}
 for name,raw in [('config.json',json.dumps(cfg).encode()),('tokenizer.json',b'CPU_MOCK_ONLY'),('tokenizer_config.json',b'CPU_MOCK_ONLY')]: (model/name).write_bytes(raw)
 files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in model.iterdir()}
 rows=[dict(id=g['id'],question=g['question'],prompt_text='CPU MOCK NOT REAL TOKENIZATION '+g['question'],input_ids=[1]*100,input_tokens=100,max_new_tokens=512,prompt_not_truncated=True,no_padding=True,gold_used_in_prompt=False)for g in gold if g['id']!=other]
 p={'cases':rows,'skipped':[{'id':other,'reason':'CPU_MOCK_EXCLUDED'}],'accepted_count':63,'planned_count':64,'planned_ids':[g['id']for g in gold],'shot_count':3,'capacity':1024,'maximum_new_tokens':512,'tokenizer_files_SHA256':files,'source_SHA256':GOLD_REMOTE_SHA}
 def invoke(value,cfg_value=cfg):
  return validate_payloads(value,{'cases':gold},model,cfg_value)
 proof=invoke(p)[2];assert proof['admitted']==63 and proof['rejected']==1 and proof['selected_count']==4
 tests=['real_common_function_accepts63plus1skip_with_same4_selected']
 for name in ['accepted_count','planned_ids','selected_rejected','question','crop','tokens','cap','tokenizer_hash','layer_geometry']:
  bad=copy.deepcopy(p);badcfg=copy.deepcopy(cfg)
  if name=='accepted_count':bad['accepted_count']=64
  elif name=='planned_ids':bad['planned_ids'][-1]='changed-question'
  elif name=='selected_rejected':
   row=next(x for x in bad['cases']if x['id']==IDS[0]);bad['cases'].remove(row);bad['accepted_count']-=1;bad['skipped'].append({'id':row['id']})
  elif name=='question':next(x for x in bad['cases']if x['id']==IDS[0])['question']='changed question'
  elif name=='crop':next(x for x in bad['cases']if x['id']==IDS[0])['prompt_not_truncated']=False
  elif name=='tokens':next(x for x in bad['cases']if x['id']==IDS[0])['input_tokens']=99
  elif name=='cap':next(x for x in bad['cases']if x['id']==IDS[0])['max_new_tokens']=511
  elif name=='tokenizer_hash':bad['tokenizer_files_SHA256']['tokenizer.json']='0'*64
  else:badcfg['compress_ratios']=[0]*44
  try:invoke(bad,badcfg)
  except AssertionError:tests.append('reject_'+name)
  else:raise AssertionError('CPU regression accepted malformed '+name)
 (root/'hash_fixture.txt').write_bytes(b'CPU_MOCK_ONLY')
 try:read_pinned(root/'hash_fixture.txt','0'*64)
 except AssertionError:tests.append('reject_pinned_bytes_SHA')
 else:raise AssertionError('SHA mismatch accepted')
 return {'passed':True,'CPU_only':True,'GPU':False,'Native_loaded':False,'mock_fixture_not_benchmark':True,'tests':tests}

if __name__=='__main__':
 import sys
 print(json.dumps(run_cases(Path(sys.argv[1]))))
