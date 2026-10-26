import sys,pathlib,json,hashlib,subprocess,os,ast,socket,time
D=pathlib.Path(__file__).resolve().parent;rt=D/'runtime';sys.path.insert(0,str(rt))
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
import target_protocol,batch_cohort_r9128,batch_native_r9128,host_gap_r9258,task_quality_r9258,quality_gate_r9258
import entry_contract_r9143,parameter_loader_r9143,native_binding_r9143
m=json.loads((rt/'model-manifest-r9183.json').read_bytes());entry_contract_r9143.validate(m)
model,receipt=entry_contract_r9143.verify_inputs(m)
z=subprocess.run(['ldd',receipt['native']],text=True,capture_output=True,timeout=60)
assert z.returncode==0 and'not found'not in z.stdout+z.stderr
assert'torch'not in sys.modules
header=pathlib.Path('/refactor/incoming/Kimi-speculative-batch-Model-r9122c/kimi_pair_state_r9006.hpp')
assert sha(header)=='ed6b0dc47b5d33a527716577f316718830fb616a8c4eec71b306f4639e0c32ef'
prior=pathlib.Path('/refactor/incoming/Kimi-relaxed-hostlogits-r9258a')
assert sha(prior/'native_terminal_contract_r9258.cpp')==sha(D/'native_terminal_contract_r9258.cpp')
native=json.loads(subprocess.check_output([str(prior/'CPU_terminal_contract')],text=True,timeout=20))
# Actual GPU static libraries deliberately not dlopened by this CPU job.
from transformers import AutoTokenizer
import torch
assert not torch.cuda.is_initialized()
tok=AutoTokenizer.from_pretrained(str(model),trust_remote_code=True,local_files_only=True)
fixed=json.loads(pathlib.Path('/refactor/incoming/r4917-Kimi-two-geometry-frozen-gold.json').read_bytes())
special=host_gap_r9258.special_provenance(tok,set(fixed['eos_token_ids']),json.loads((model/'config.json').read_bytes())['text_config']['vocab_size'])
data=json.loads((D/'frozen-quality-data-r9258.json').read_bytes());tokenized=[];over=[]
for phase in ['pilot8','heldout96']:
 for q in data[phase]:
  text,meta=task_quality_r9258.prepare(q)
  other,_=task_quality_r9258.prepare(dict(q,answer='gold_metadata_probe_must_not_appear_918262'))
  assert text==other
  ids=tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True)
  assert ids==tok.apply_chat_template([{'role':'user','content':other}],tokenize=True,add_generation_prompt=True)
  assert all(0<=int(x)<special['model_vocab_size']for x in ids)
  row={'phase':phase,'id':q['id'],'domain':q['domain'],'input_tokens':len(ids),'cap':512,'fits':1<=len(ids)<=512 and len(ids)+512<=1024,'actual_input_IDs_within_ModelV':True,'gold_metadata_independent_prompt_and_tokens':True,'input_ids_SHA256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()}
  tokenized.append(row)
  if not row['fits']:over.append(row)
# Do not drop/truncate failures: fixeddata remain frozen; admission must explicitly reject/exclude as coverage.
assert not torch.cuda.is_initialized()
cases=[{'id':'x','domain':'a','status':'completed','correct':True}]
assert quality_gate_r9258.paired_gate(cases,cases)['stop']is False
assert quality_gate_r9258.paired_gate(cases,[dict(cases[0],correct=False)])['stop']is True
out={'passed':True,'CPU_only':True,'GPU':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'hostname':socket.gethostname(),'no_native_dlopen_or_Runtime':True,'native_DSO_SHA256':m['native_DSO_SHA256'],'manifest_SHA256':sha(rt/'model-manifest-r9183.json'),'actual_source_DSO_objects_ldd_closure':True,'linked_terminal_header_SHA256':sha(header),'actual_header_CPU_metadata':native,'special_provenance':special,'tokenizer_file_SHA256':{p.name:sha(p)for p in model.glob('*token*')if p.is_file()and p.suffix=='.json'},'tokenized':tokenized,'overcapacity_preserved':over,'all104fit':not over,'all104_input_IDs_valid':True,'all_pilot8_fit':not any(v['phase']=='pilot8'for v in over),'CUDA_initialized':torch.cuda.is_initialized(),'GPU_math_or_Model_quality_or_speed_qualified':False}
pathlib.Path('/refactor/results/Kimi-relaxed-CPU-r9258b.json').write_bytes(json.dumps(out,indent=2).encode());print(json.dumps({'passed':True,'actual_step':out['actual_step'],'noGPU':True,'all104fit':not over,'all104_input_IDs_valid':True,'all_pilot8_fit':not any(v['phase']=='pilot8'for v in over),'over':len(over),'special_count':len(special['protected_ids'])}),flush=True)
