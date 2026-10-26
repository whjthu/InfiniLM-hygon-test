import pathlib,sys,json,hashlib,ast,ctypes as C,time
import numpy as np,copy
D=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(D/'runtime'));old=D.parent/'Kimi-W2-pair-Model-private-r9282';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();start=time.perf_counter()
from entry_contract_r9143 import validate,geometry
from batch_native_r9128 import Native
from batch_cohort_r9128 import Request,cohort
from adaptive_width_r9223 import width_cap
m=json.loads((D/'runtime/model-manifest-r9183.json').read_bytes());assert validate(m)=={'tp_nodes':2,'pp':2,'logical_capacity':8};assert geometry(8)=={'max_rows':16,'pages':68,'page_size':256,'KDA_pool_rows':17,'pages_per_request':4,'max_total_tokens':1024}
for key,h in m['source_SHA256'].items():
 if key.startswith('/refactor/incoming/Kimi-cap8-micro1-Model-r9293/runtime/'):assert sha(D/'runtime'/key.split('/')[-1])==h,key
same=['batch_native_r9128.py','batch_policy_r9128.py','parameter_loader_r9143.py','W2_policy_r9282.py','batch_frames_r9142.py','draft_context_r9128.py','batch_draft_worker_r9202.py','entry_contract_r9143.py','memory_guard_r9144.py','guard_snapshot_r9225.py']
for name in same:assert sha(D/'runtime'/name)==sha(old/'runtime'/name)
a=(old/'runtime/batch_cohort_r9128.py').read_bytes().decode();a=a.replace('range(0,len(requests),2)','range(0,len(requests),1)').replace('group=requests[begin:begin+2]','group=requests[begin:begin+1]').replace("'prefill_micro_request_cap':2","'prefill_micro_request_cap':1");assert a==(D/'runtime/batch_cohort_r9128.py').read_bytes().decode()
for p in D.rglob('*'):
 if p.suffix in ['.py','.sh']:assert b'\r'not in p.read_bytes()
 if p.suffix=='.py':ast.parse(p.read_bytes())
data=json.loads((D/'frozen-data-r9293.json').read_bytes());proof={v['id']:v for v in data['prompt_proof']};commands=[json.loads(p.read_bytes())for p in sorted((D/'queue').glob('q*.json'))];cases=[c for c in commands if c['action']=='case'];assert len(cases)==50 and commands[-1]['action']=='stop'
for c in cases:
 assert 1<=c['C']<=8 and len(c['requests'])==c['C']and c['W']==8 and c['C']*8<=64 and c['cap']in[8,512]and c['epsilon']==0 and not c['draft_projected_cache']
 for q in c['requests']:pr=proof[q['id']];assert pr['admitted']and pr['all_model_IDs_valid']and pr['input_tokens']+512<=1024
main=[c for c in cases if c['category']=='main-C2'];assert len(main)==32
for g in range(16):
 a,b=main[2*g:2*g+2];assert a['requests']==b['requests']and a['W2pair']!=b['W2pair']and a['W2pair']==bool(g%2)
assert len({q['id']for c in main for q in c['requests']})==32 and sorted(c['C']for c in cases[-8::2])==[7,8,8,8]
source=D.parent/'Kimi-speculative-batch-private-r9122c/pair_probe.cpp';ss=source.read_bytes().decode();assert sha(source)=='c704a99e0d261ded3e185a18f39ac3e1ffe14a3c53ced0f9f9ed028c3a1f96fa';assert '(logical!=8&&logical!=16&&logical!=32)'in ss and 'c->pair_r9006->geometry.logical!=c->max_rows/2'in ss
for n in [1,2,4,8]:assert width_cap(n)==8 and n*width_cap(n)<=64
class Fn:
 def __init__(self,f):self.f=f
 def __call__(self,*a):return self.f(*a)
def arr(p,typ,n):return np.ctypeslib.as_array(C.cast(p,C.POINTER(typ)),shape=(n,))
class Lib:
 def __init__(self):
  self.calls=[]
  for name in ['reset','mode','prefill','forward','commit','replay','rows','stats','taps_begin','taps_count','taps_info','taps_copy']:setattr(self,'modern_kimi_batch_'+name+'_r9122',Fn(lambda *a:0))
  self.modern_kimi_batch_capabilities_r9122=Fn(lambda ctx:json.dumps({'implemented_max_flatten_tokens':64,'logical_capacity':8,'implemented_max_requests':32,'native_batch':True,'pool_rows':17,'flags':3}).encode());self.modern_kimi_batch_taps_capabilities_r9122=Fn(lambda:0x01200500);self.modern_kimi_batch_prefill_r9122=Fn(self.prefill)
 def prefill(self,ctx,T,n,ids,off,tokens,pos,logits,out,flags):
  assert n==1;logical=arr(ids,C.c_int32,n).tolist();offsets=arr(off,C.c_int32,n+1).tolist();positions=arr(pos,C.c_int64,T).tolist();flat=arr(tokens,C.c_int64,T).tolist();assert logical[0]in range(1,9)and offsets==[0,T]and positions==list(range(T))and flat==list(range(T));arr(logits,C.c_uint16,n*128)[:]=0;arr(out,C.c_int64,n)[:]=7;arr(flags,C.c_int32,1)[:]=3;self.calls.append({'logical':logical,'T':T,'offsets':offsets,'positions_values_preserved':True});return 0
class CT:
 def collect(self,k,x):return[dict(x,selected=None),dict(x,selected=None),x,dict(x,selected=None)]
lib=Lib();nn=Native({'lib':lib,'ctx':1,'control':CT(),'node':2,'tail':True,'emitter':2,'vocab':128,'pp':2});harvest=[]
def harvest_spy(ids,past,width):harvest.append({'ids':ids,'past':past,'width':width});return{k:b'feature-spy'for k in ids}
nn.harvest=harvest_spy
for key,T in zip(range(1,9),[159,409,512,1,128,7,311,42]):
 seen=[];sel,features,record=nn.call('prefill',[key],[0],[list(range(T))],True,lambda x:seen.extend(x));assert sel==seen==[7]and record['width']==[T]and features=={key:b'feature-spy'}
assert len(lib.calls)==len(harvest)==8
class NC:
 maxC=8;maxT=64;card_count=33;observer=None
 def __init__(self):self.calls=[];self.guards=[];self.memory_guard=lambda label:self.guards.append(label)
 def reset(self,state):pass
 def I192(self):return[[0,0]for _ in range(8)]
 def stats(self):return[0]*64
 def call(self,kind,ids,past,tokens,capture,callback):
  assert kind=='prefill'and len(ids)==1 and past==[0];callback([127]);rec={'kind':kind,'logical':list(ids),'width':list(map(len,tokens)),'CAPI_maxhost_s':0.,'feature_harvest_local_s':0.};self.calls.append(rec);return[127],None,rec
class CC:
 def collect(self,key,value):return[value]*4
results=[]
for Cn in [2,8]:
 n=NC();qs=[Request(k,list(range(159 if k==1 else 409 if k==2 else 17+k)),8,1,{'answer':'0'})for k in range(1,Cn+1)];out=cohort(n,CC(),lambda *a,**k:None,2,2,qs,'cpu-C'+str(Cn),8,'state',True,1,{127},time.monotonic()+30,adaptive_width=True);assert len(n.calls)==Cn and [k for x in n.calls for k in x['logical']]==list(range(1,Cn+1))and all(q.done for q in qs)and out['prefill_micro_request_cap']==1;assert len(n.guards)==Cn
 results.append({'C':Cn,'actualcohort_prefill_calls':Cn,'maxT':max(v['width'][0]for v in n.calls),'allEOS':True,'guard_aftereachmicro':True})
from service_metrics_r9282 import screen_gate
screen=[]
for g in range(4):
 for arm in [False,True]:screen.append({'label':'main-C2-g%02d-%d'%(g,int(arm)),'configured_C':2,'command':{'category':'main-C2','W2pair':arm},'request_metrics':[{'quality':{'id':'fixed'+str(2*g+j),'correct':True},'EOS':j==0,'truncated':j==1}for j in range(2)]})
assert screen_gate(screen[:7])is None and not screen_gate(screen)['stop']
wrong=copy.deepcopy(screen);wrong[1]['request_metrics'][0]['quality']['correct']=False;wrong[3]['request_metrics'][0]['quality']['correct']=False;assert screen_gate(wrong)['stop']
nonEOS=copy.deepcopy(screen);nonEOS[1]['request_metrics'][0]['EOS']=False;nonEOS[3]['request_metrics'][0]['EOS']=False;assert screen_gate(nonEOS)['stop']
one=copy.deepcopy(screen);one[1]['request_metrics'][0]['quality']['correct']=False;assert not screen_gate(one)['stop']
report={'passed':True,'CPU_only':True,'GPU':False,'Runtime_created':False,'native_dlopen':False,'new_steps':0,'manifest_SHA256':sha(D/'runtime/model-manifest-r9183.json'),'actual_Native_Python_prefill_ctypes_pointer_calls':lib.calls,'feature_harvest_spy_metadata':harvest,'actualcohort_python_mock_tests':results,'unchanged_transaction_and_draft_sources_SHA_reused':{n:sha(D/'runtime'/n)for n in same},'native_geometry_source_SHA256':sha(source),'geometry':geometry(8),'logical_active_bank':list(range(1,9)),'shadow_bank':list(range(9,17)),'physicalfloor':16,'fixed_cases':50,'first8_mainC2_actualscreenrule_checks_passed':True,'C2_unique':32,'MMLU31_admitted':True,'sharedcapacityreject11704':True,'tokens_proof_reused':'910406.275','new_tokenization':False,'scope':'actualPythonABIbuffers/offsets/cohortmicro/EOS; neuralmath+CAPI mocked, notHIP/nativeRuntimeorGPUmemqualification','pending_actual_stage_environment_imports_ldd_bashn':True,'elapsed_s':time.perf_counter()-start};p=D/'CPU-contract-r9293.json';p.write_bytes((json.dumps(report,indent=2)+'\n').encode());print(json.dumps({k:report[k]for k in ['passed','CPU_only','manifest_SHA256','actualcohort_python_mock_tests','fixed_cases','elapsed_s']}))
