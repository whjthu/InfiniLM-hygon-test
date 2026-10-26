import pathlib,json,sys,os,time,hashlib,subprocess,ast,copy
D=pathlib.Path(__file__).resolve().parent;r=pathlib.Path('/refactor');sys.path.insert(0,str(D/'runtime'));sys.path.insert(0,str(D));sys.path.append('/refactor/incoming');start=time.monotonic();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for p in D.rglob('*'):
 if p.suffix in ['.py','.sh']:assert b'\r'not in p.read_bytes(),str(p)
 if p.suffix=='.py':ast.parse(p.read_bytes())
 if p.suffix=='.sh':assert subprocess.run(['bash','-n',str(p)],capture_output=True).returncode==0,str(p)
import entry_contract_r9143,target_protocol,native_binding_r9143,parameter_loader_r9143
m=json.loads((D/'runtime/model-manifest-r9183.json').read_bytes());entry_contract_r9143.validate(m);model,b=entry_contract_r9143.verify_inputs(m)
original=json.loads(pathlib.Path(b['original_CPU_receipt']).read_bytes());assert original['completed']and not original['GPU_execution']and original['native_SHA256']==b['native_SHA256']and original['all_linked_objects']==b['all_linked_object_SHA256'];assert original['actual_step']=='910407.427'
from transformers import AutoTokenizer
from task_quality_r9258 import prepare
import torch
assert not torch.cuda.is_initialized()
tok=AutoTokenizer.from_pretrained(str(model),trust_remote_code=True);cfg=json.loads((model/'config.json').read_bytes());V=cfg['text_config']['vocab_size'];data=json.loads((D/'planned-quality-r9282.json').read_bytes());admitted=[];rejects=[];proof=[]
for item in data['GSM64']+data['MMLU32']:
 text,metadata=prepare(item);q=copy.deepcopy(item);q['answer']='987654321';assert prepare(q)[0]==text,'goldleak'
 ids=tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True);assert ids and all(type(x)is int and 0<=x<V for x in ids)
 fit=1<=len(ids)<=512 and len(ids)+512<=1024;proof.append({'id':item['id'],'input_tokens':len(ids),'output_cap':512,'all_model_IDs_valid':True,'admitted':fit,'prompt_SHA256':hashlib.sha256(text.encode()).hexdigest(),'ids_SHA256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
 if fit:admitted.append(item['id'])
 else:rejects.append({'id':item['id'],'reason':'originalprompt+cap512 exceedsctx1024 or input512; no truncation/replacement','input_tokens':len(ids),'shared_botharms':True})
assert all(x['id']in admitted for x in data['GSM64'][:8]),'screen8 invalid'
from matrix_data_r9282 import make_commands
commands=make_commands(data,set(admitted));assert len(commands)==38
# Strict original greedy/cohort code, shared-emitter arrival and frozen screen gates.
from service_metrics_r9282 import service_metrics,screen_gate
parts=[]
for k in range(2):
 us=[{'first_ready_monotonic':10+100*k,'last_ready_monotonic':20+100*k,'completion_monotonic':21+100*k,'useful_tokens':8,'quality':{'correct':True},'EOS':True,'truncated':False}for _ in range(16)]
 parts.append({'request_metrics':us,'admission_native_monotonic':5+100*k,'label':str(k),'decode_common_window_s':11,'total_useful_decode_tokens':112,'aggregate_decode_tok_s':112/11})
service=service_metrics(parts,0);assert service['request_metrics'][16]['service_TTFT_s']==110 and service['service_output_tok_s']==256/121 and not service['actual_native_C32']
cases=[{'command':{'category':'screen','W2pair':bool(k)},'request_metrics':[{'quality':{'correct':True},'EOS':i<6,'truncated':i>=6}]}for k in range(2)for i in range(8)];assert not screen_gate(cases)['stop'];cases[8]['request_metrics'][0]['quality']['correct']=False;cases[9]['request_metrics'][0]['quality']['correct']=False;assert screen_gate(cases)['stop']
import tempfile,contextlib
from parameter_loader_r9143 import load_parameters
class Weight:
 shape=[1];dtype='f32'
 def data_ptr(self):return 4096
 def numel(self):return 1
 def element_size(self):return 4
 def contiguous(self):return self
class Stats:
 def __init__(self,owner):self.owner=owner
 def __call__(self,out,n):
  assert n==5;self.owner.statreads+=1
  for i,v in enumerate(self.owner.stats):out[i]=v
  return 0
class FakeLib:
 def __init__(self):self.stats=[0]*5;self.statreads=0;self.loads=0;self.finishes=0;self.modern_cpu_copy_stats_r2515=Stats(self)
 def modern_kimi_engine_load_r2475(self,ctx,payload):
  assert len(json.loads(payload))==3;self.loads+=1;self.stats=[1,3,12,0,0];return 0
 def modern_kimi_engine_finish_r2475(self,ctx):self.finishes+=1;return 0
class FakeTorch:float32='f32';bfloat16='bf16';uint8='u8'
class Shard:
 def get_tensor(self,name):return Weight()
@contextlib.contextmanager
def open_mock(*args,**kwargs):yield Shard()
with tempfile.TemporaryDirectory()as tmp:
 modeldir=pathlib.Path(tmp);(modeldir/'model.safetensors.index.json').write_bytes(json.dumps({'weight_map':{k:'single.safetensors'for k in ['a','b','c']}}).encode());fake=FakeLib();loaded=load_parameters(fake,0,modeldir,{},set(['a','b','c']),open_mock,FakeTorch,lambda rc:None,lambda x:None);assert fake.loads==fake.finishes==1 and fake.statreads==2 and loaded['native_load_flush_profiles'][0]['r2515_delta']==[1,3,12,0,0]and loaded['extra_empty_load_calls']==0
assert not torch.cuda.is_initialized();parents={}
parents=json.loads((D/'parent-budget-snapshot.json').read_bytes());assert set(parents)=={'910379','910371','910406'} and all(v['JobState']=='RUNNING'for v in parents.values())
Q=D/'queue';Q.mkdir(exist_ok=True)
for i,command in enumerate(commands):
 p=Q/('q%04d.json'%i);raw=(json.dumps(command,indent=2)+'\n').encode();assert not p.exists();p.write_bytes(raw)
stop=Q/'q9999.json';assert not stop.exists();stop.write_bytes((json.dumps({'action':'stop','reason':'frozenmatchedmatrixcomplete'})+'\n').encode())
report={'passed':True,'CPU_only':True,'GPU':False,'Runtime_created':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'host':os.uname().nodename,'manifest_SHA256':sha(D/'runtime/model-manifest-r9183.json'),'all96_input_IDs_valid':True,'all_screen8_fit':True,'planned_unique':96,'admitted_unique':len(admitted),'admitted_ids':admitted,'rejections':rejects,'prompt_proof':proof,'gold_not_in_prompt':True,'C32_samearrival_contract_passed':True,'perflush_counter_mock_same_native_call_sequence':True,'equalcaps_not_automatic_screen_veto':True,'screen_excess2_rule_passed':True,'parent_snapshot':parents,'preparation_s':time.monotonic()-start,'commands':commands,'queue_SHA256':{p.name:sha(p)for p in Q.iterdir()}}
(r/'results/Kimi-W2pair-CPU-r9282.json').write_bytes((json.dumps(report,indent=2)+'\n').encode());print(json.dumps({k:report[k]for k in ['passed','actual_step','manifest_SHA256','admitted_unique','rejections','preparation_s']}))
