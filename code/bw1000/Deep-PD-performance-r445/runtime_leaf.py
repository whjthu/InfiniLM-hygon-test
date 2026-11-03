"""The original forward/loader namespace plus the complete state-codec CAPI."""
from pathlib import Path
import copy,hashlib,json,time,ctypes,os
from native_codec import Codec,ABI

def encode_once(provider,plan,root,run,role):
 from infinilm.server.agent_protocol import prepare_agent_request
 engine=provider['prepared']['engine'];assert not engine._requests
 inputs={}
 for key,spec in plan['fixtures'].items():
  path=Path(spec['path']);raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==spec['SHA256']
  fixture=json.loads(raw);body=fixture['body']if key=='L'else fixture
  data,schemas,choice=prepare_agent_request(copy.deepcopy(body),'deepseek_v4');data.update(_agent_schemas=schemas,_agent_choice=choice)
  prompt,ids=engine.encode_chat(data['messages'],data,True,data.get('chat_template_kwargs')or{});ids=list(ids)
  digest=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
  assert len(ids)==spec['actual_prompt_tokens']
  if spec['actual_prompt_token_ids_SHA256']is not None:assert digest==spec['actual_prompt_token_ids_SHA256']
  assert not engine._requests
  target=Path(root)/'results'/('Deep382-'+run+'-'+role+'-'+key+'-private-prompt-IDs.json');assert not target.exists()
  target.write_text(json.dumps(ids,separators=(',',':'))+'\n')
  inputs[key]=dict(ids=ids,fixture=fixture,fixture_SHA256=spec['SHA256'],actual_prompt_tokens=len(ids),prompt_IDs_SHA256=digest,private_IDs_path=str(target),private_IDs_file_SHA256=hashlib.sha256(target.read_bytes()).hexdigest(),prompt_render_SHA256=hashlib.sha256(prompt.encode()).hexdigest(),single_runtime_encode=True,no_service_request_enqueued=True)
 provider['PD_inputs']=inputs


def type_equal(actual,expected):
 if type(actual)is not type(expected):return False
 if isinstance(expected,dict):return actual.keys()==expected.keys()and all(type_equal(actual[k],v)for k,v in expected.items())
 if isinstance(expected,list):return len(actual)==len(expected)and all(type_equal(a,b)for a,b in zip(actual,expected))
 return actual==expected

class NamespaceLeaf:
 actual_Native=True
 def __init__(self,ns,provider,wire,pin):
  self.ns,self.provider,self.wire=ns,provider,wire
  self.inputs=provider['PD_inputs'];self.reasons=provider['prepared']['FinishReason'];self.check=ns['check'];self.forward=ns['forward']
  from worker_r206b import owner_threads
  self.owner_observation=owner_threads(ns)
  self.receipt=dict(Native_SHA256=pin['Native_SHA256'],rows=1,capacity=32768,TP=8,PP=1,keys=ns['report']['keys'],provider=provider['proof'],inputs={name:{k:v for k,v in value.items()if k not in ['ids','fixture']}for name,value in self.inputs.items()},owner_observation=self.owner_observation,legacy_loader_phase=ns['report'].get('phase'),load_seconds=ns['report'].get('loading_seconds'),actual_Native=True)
  model_identity=wire.sha(wire.canonical(dict(config_SHA256=hashlib.sha256((ns['model']/'config.json').read_bytes()).hexdigest(),index_SHA256=hashlib.sha256((ns['model']/'model.safetensors.index.json').read_bytes()).hexdigest())))
  abi_identity=wire.sha(wire.canonical(dict(version=ABI,header_SHA256=pin['ABI_header_SHA256'],Native_SHA256=pin['Native_SHA256'])))
  self.codec=Codec(ns['lib'],ns['ctx'],ns['check'],wire,model_identity,abi_identity)
  props=[ns['torch'].cuda.get_device_properties(i)for i in range(8)]
  self.receipt['actual_devices']=[dict(logical_device=i,name=p.name,total_memory=p.total_memory,gcnArchName=getattr(p,'gcnArchName',None),PCI_bus_id=getattr(p,'pci_bus_id',None))for i,p in enumerate(props)]
  assert all('gfx936'in p['gcnArchName']for p in self.receipt['actual_devices'])
  owners=ns['report']['owner_affinity_before_weight_load']['actual_owner_report']['owners'];masks=[set(map(int,o['verified_mask'].split(',')))for o in owners]
  assert len(masks)==8 and all(len(mask)==4 for mask in masks)and sum(map(len,masks))==len(set().union(*masks))==32
  assert set().union(*masks)<=set(os.sched_getaffinity(0))
  self.receipt['owner_affinity_8_disjoint_masks']=ns['report']['owner_affinity_before_weight_load']
  providers={}
  for name,fn in self.codec.fn.items():
   info=ns['DeepDlInfo']();assert ns['dla'](ctypes.cast(fn,ctypes.c_void_p),ctypes.byref(info))
   path=Path(info.fname.decode()).resolve();assert path==Path(ns['artifact']).resolve()and hashlib.sha256(path.read_bytes()).hexdigest()==pin['Native_SHA256']
   providers[name]=dict(path=str(path),SHA256=pin['Native_SHA256'])
  self.receipt['actual_PD_symbol_providers']=providers
  # All arms keep the same actual defaultOFF epilogue and existing forward
  # policies. Do not inherit a parent's Model qualification for this new DSO.
  lib=ns['lib'];lib.deep_hc_model_epilogue_get_r366.argtypes=[];lib.deep_hc_model_epilogue_get_r366.restype=ctypes.c_int
  assert lib.deep_hc_model_epilogue_get_r366()==0
  from hc_model_contract import HCModel
  self.receipt['fixed_HC_loaded_instances']=HCModel(lib,ns['cfg']['num_hidden_layers']).initial
  self.receipt['fixed_HC_epilogue_mode']=0
 def reset(self):
  self.check(self.ns['lib'].modern_native_engine_reset(self.ns['ctx']))
  memory=[list(self.ns['torch'].cuda.mem_get_info(i))for i in range(8)]
  if min(v[0]for v in memory)<16<<30:raise MemoryError('Inherited16GiB/card request admission floor')
  self.last_memory_checkpoint=memory
 def stage_started(self,operation,meta):self.current_stage=(operation,meta['epoch'])
 def finite(self,logits):
  np=self.ns['np'];row=(logits[0].astype(np.uint32)<<16).view(np.float32)
  if not np.isfinite(row).all():raise RuntimeError('Nonfinite actual Native logits')
 def begin_export(self,*args):return self.codec.begin_export(*args)
 def prepare_import(self,*args):return self.codec.prepare_import(*args)
 def prompt_for(self,meta):
  value=self.inputs[meta['fixture']];assert len(value['ids'])==meta['past']and value['prompt_IDs_SHA256']==meta['prompt_IDs_SHA256'];return value['ids']
 def quality(self,ids,meta):
  from content_json import parse_final_object
  text=self.ns['tokenizer'].decode([v for v in ids if v!=1],skip_special_tokens=False,clean_up_tokenization_spaces=False)
  try:
   fixture=self.inputs[meta['fixture']]['fixture']
   if meta['fixture']=='S':
    import re
    matches=re.findall(r'####\s*([ABCD])\b',text);value=ord(matches[-1])-65 if matches else None;presentation='Original MMLU last #### letter parser'
    correct=value==0
   else:
    value,presentation=parse_final_object(text);correct=type_equal(value,fixture['expected_answer'])
   return dict(task_correct=correct,content=text,presentation=presentation,scope='Two fixed historical performance shapes; not broad quality qualification')
  except Exception as e:return dict(task_correct=False,content=text,parse_error=repr(e),scope='Failed answer retained, no request retry')
 def close_python_provider(self):
  engine=self.provider['prepared']['engine'];assert not engine._requests;engine.stop()
