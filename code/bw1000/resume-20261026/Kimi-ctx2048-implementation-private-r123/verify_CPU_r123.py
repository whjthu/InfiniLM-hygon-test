import pathlib,sys,json,hashlib,ast,types,ctypes as C,socket,threading,time,copy,subprocess
import numpy as np
D=pathlib.Path(__file__).resolve().parent;R=D.parent;P=D/'runtime';N=D/'native';sys.path.insert(0,str(P));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();start=time.perf_counter()
from entry_contract_r9143 import validate,geometry
from draft_context_r9128 import ContextStore
from committed_context_r9202 import CommittedFeatureContexts
from batch_frames_r9142 import send_query,receive_bodies,FEATURE_BYTES
from batch_wire_r9128 import recv,MAX_BODY
from batch_native_r9128 import Native
import target_protocol
assert geometry(8)=={'max_rows':16,'pages':132,'page_size':256,'KDA_pool_rows':17,'pages_per_request':8,'max_total_tokens':2048}
m=json.loads((P/'model-manifest-r9183.json').read_bytes());assert validate(m)=={'tp_nodes':2,'pp':2,'logical_capacity':8};assert m['native_DSO_SHA256']=='PENDING_ACTUAL_HOST_BUILD_R123';assert MAX_BODY==128<<20 and FEATURE_BYTES==71680
# FrozenNativeAPIexistsatstartup, cannotfallbacktoolddso.
assert 'modern_kimi_engine_create_ctx2048_r123'in(P/'native_binding_r9143.py').read_bytes().decode()
checks=0
for row in range(1,17):
 table=list(range(1+(row-1)*8,1+row*8));assert max(table)<132
 for pos in [0,255,256,1023,1024,2047]:assert(table[pos//256]*256+pos%256)<132*256;checks+=1
# Actualnewfrontendboundedfields, noTokenizer/goldchange.
tree=ast.parse((P/'target_protocol.py').read_bytes());fn=next(x for x in ast.walk(tree)if isinstance(x,ast.FunctionDef)and x.name=='execute');tr=next(x for x in fn.body if isinstance(x,ast.Try));prefix=tr.body[:next(i for i,x in enumerate(tr.body)if isinstance(x,ast.For))];code=compile(ast.fix_missing_locations(ast.Module(body=prefix,type_ignores=[])),'frozennewfrontend','exec');native=types.SimpleNamespace(maxC=8,maxT=64)
def command(cap):return{'schema':1,'action':'case','id':'cpu','C':2,'W':8,'strategy':'state','I192':True,'adaptive_width':True,'W2pair':True,'epsilon':0,'draft_projected_cache':False,'cap':cap,'case_budget_s':180,'requests':[{},{}]}
exec(code,{'command':command(1024),'native':native});checks+=1
try:exec(code,{'command':command(1025),'native':native})
except AssertionError:checks+=1
else:raise AssertionError('cap1025accepted')
proof=json.loads((R/'Kimi-cap8-micro1-Model-private-r9293/frozen-data-r9293.json').read_bytes());ps={q['id']:q for q in proof['prompt_proof']}
for q in proof['MMLU_planned32']:v=ps[q['id']];assert v['all_model_IDs_valid']and 1<=v['input_tokens']<=1024 and v['input_tokens']+1024<=2048;checks+=1
assert ps['mmlu-test-11704']['input_tokens']==576
# Actualcommittedstore: oneinitial1024 then128committed8rowupdates ->2048; atomicrejections/retire/epoch.
store=ContextStore(8);store.begin(1);store.append(1,1,0,1024,b'A'*1024,lambda a,b:a+b)
for past in range(1024,2048,8):store.append(1,1,past,8,b'B'*8,lambda a,b:a+b)
assert store.values[(1,1)][0]==2048 and len(store.values[(1,1)][1])==2048;before=store.values.copy()
for args in [(1,1,2048,1),(1,1,1024,1),(2,1,2048,1)]:
 try:store.append(*args,b'x',lambda a,b:a+b)
 except AssertionError:assert store.values==before;checks+=1
 else:raise AssertionError('badpast/epoch/contextaccepted')
store.retire(1,1);assert not store.values
try:store.append(1,1,0,1,b'x',lambda a,b:a+b)
except AssertionError:checks+=1
else:raise AssertionError('retiredrequestrevived')
# Preserve strictnoninitialcommittedprefix<=8 evenrawstoremaxappend1024.
c=object.__new__(CommittedFeatureContexts);c.owner=threading.get_ident();c.raw=ContextStore(8);c.raw.begin(1);c.raw.append(1,1,0,1024,b'A',lambda a,b:a+b);c.enabled=False
try:c.validate_many(1,[{'request':1,'past':1024,'rows':9}])
except AssertionError:checks+=1
else:raise AssertionError('uncommittedsuffixaccepted')
# Realbinarysocket70MiBprefillframe crossingold64MiBboundary, noactivationdata persisted.
a,b=socket.socketpair();a.settimeout(20);b.settimeout(20);errors=[];result={}
def receiver():
 try:
  meta,body=recv(b);assert not body and meta['cmd']=='query_batch';parts=receive_bodies(b,meta);assert len(parts)==1 and len(parts[0])==73400320;result['bytes']=len(parts[0]);result['first_last_zero']=parts[0][:2]==parts[0][-2:]==b'\0\0'
 except BaseException as e:errors.append(repr(e))
t=threading.Thread(target=receiver);t.start();body=bytes(1024*FEATURE_BYTES);send_query(a,{'cmd':'query_batch','seq':0,'requests':[{'request':1,'past':0,'rows':1024,'anchor':17,'K':7}]},[body]);t.join(25);assert not t.is_alive()and not errors and result['bytes']==len(body)
try:send_query(a,{'cmd':'query_batch','seq':1,'requests':[{'request':1,'past':0,'rows':1025}]},[body])
except AssertionError:checks+=1
else:raise AssertionError('rows1025accepted')
a.close();b.close();del body
# RealNativePythonctypes buffers withfakeCAPI; no native dlopen/Runtime/math execution.
class F:
 def __init__(self,fn):self.fn=fn
 def __call__(self,*args):return self.fn(*args)
def view(p,typ,n):return np.ctypeslib.as_array(C.cast(p,C.POINTER(typ)),shape=(n,))
class L:
 def __init__(self):
  self.records=[]
  for k in ['reset','mode','prefill','forward','commit','replay','rows','stats','taps_begin','taps_count','taps_info','taps_copy']:setattr(self,'modern_kimi_batch_'+k+'_r9122',F(lambda *x:0))
  self.modern_kimi_batch_capabilities_r9122=F(lambda ctx:json.dumps({'native_batch':True,'logical_capacity':8,'implemented_max_requests':32,'implemented_max_flatten_tokens':64,'pool_rows':17,'flags':3,'pages_per_request':8,'max_total_tokens':2048}).encode());self.modern_kimi_batch_taps_capabilities_r9122=F(lambda:0x01200500);self.modern_kimi_batch_prefill_r9122=F(self.prefill);self.modern_kimi_batch_forward_r9122=F(self.forward);self.modern_kimi_batch_rows_r9122=F(self.rows)
 def rows(self,ctx,n,ids,out):view(out,C.c_int32,n*4)[:]=[1,9,2040,1];return 0
 def output(self,n,logits,out,flags):view(logits,C.c_uint16,n*128)[:]=0;view(out,C.c_int64,n)[:]=7;view(flags,C.c_int32,1)[:]=3;return 0
 def prefill(self,ctx,T,n,ids,offset,flat,pos,logits,out,flags):
  assert T==1024 and n==1 and view(offset,C.c_int32,2).tolist()==[0,1024]and view(pos,C.c_int64,T).tolist()==list(range(1024));self.records.append({'kind':'prefill','T':T,'positions':[0,1023]});return self.output(n,logits,out,flags)
 def forward(self,ctx,T,n,fl,read,write,ids,past,total,offset,flat,pos,logits,out,flags):
  assert T==8 and n==1 and fl==3 and view(past,C.c_int32,1)[0]==2040 and view(total,C.c_int32,1)[0]==2048 and view(pos,C.c_int64,T).tolist()==list(range(2040,2048));self.records.append({'kind':'verify','T':T,'positions':[2040,2047]});return self.output(T,logits,out,flags)
class Control:
 def collect(self,key,value):return[dict(value,selected=None),dict(value,selected=None),value,dict(value,selected=None)]
lib=L();nn=Native({'lib':lib,'ctx':1,'control':Control(),'node':2,'tail':True,'emitter':2,'vocab':128,'pp':2});nn.call('prefill',[1],[0],[list(range(1024))],False);nn.call('verify',[1],[2040],[list(range(8))],False);assert len(lib.records)==2
# Sourcepin/legacyguardintent and ABIheaders, no blanket1024replacement.
h=(N/'kimi_engine_input_r145.hpp').read_bytes().decode();assert 'freeze_input_r123'in h and 'pages_per_row!=4&&pages_per_row!=3&&pages_per_row!=8'in h and 'spec&&pages_per_row!=4&&pages_per_row!=8'in h
assert 'cache_pages_r4974('not in h and 'rows==16&&c->num_blocks()==8*rows+4'in h
assert 'num_blocks() / 4'in(N/'kimi_compact_allocate_r4974.cpp').read_bytes().decode()and '(compact_capability||ctx2048_r123)'in(N/'kimi_compact_allocate_r4974.cpp').read_bytes().decode()
for name in ['engine/infer_engine.hpp','engine/rank_worker.hpp','rank_worker.hpp']:assert sha(N/name)==sha(R/'Kimi-speculative-batch-private-r9122c'/name)
for p in D.rglob('*.py'):ast.parse(p.read_bytes());assert b'\r'not in p.read_bytes()
assert 'torch'not in sys.modules
from service_metrics_r9282 import screen_gate
screens=[]
for arm in [False,True]:screens.append({'configured_C':8,'command':{'quality_screen':True,'W2pair':arm},'request_metrics':[{'quality':{'id':'fixed'+str(i),'correct':True},'EOS':i<7,'truncated':i==7}for i in range(8)]})
assert screen_gate(screens[:1])is None and not screen_gate(screens)['stop']
changed=copy.deepcopy(screens);changed[1]['request_metrics'][0]['quality']['correct']=False;changed[1]['request_metrics'][1]['quality']['correct']=False;assert screen_gate(changed)['stop']
assert len(screen_gate(screens)['matched_quality_ids'])==8
queue=[json.loads(p.read_bytes())for p in sorted((D/'queue').glob('q*.json'))];main=[c for c in queue if c.get('category')=='ctx2048-MMLU32'];assert len(main)==8 and all(c['cap']==1024 and c['C']==8 for c in main);assert len({q['id']for c in main for q in c['requests']})==32;assert sum(c['quality_screen']for c in main)==2
out={'schema':1,'passed':True,'CPU_only':True,'compiled':False,'uploaded':False,'GPU':False,'native_dlopen':False,'Runtime_created':False,'new_Slurm_steps':0,'checks':checks,'fixed_MMLU32_queue':True,'first8_matched_gate_passed':True,'actual_newPython_Native_buffers':lib.records,'real_socket_frame':result,'rawcontext2048_epoch_retire_atomicreject_pass':True,'strict_committed_prefix8_remains':True,'all32MMLU_eligible_withouttruncation':True,'11704inputtokens':576,'actual_Cpp_State_payload_contract':'SOURCE_PREPARED_NOT_COMPILED; rootHOSTbuildrequired','actual_GPUallocator_footprint':'UNKNOWN','manifest_native_pending_failclosed':True,'header_class_layout_sources_unchanged':True,'elapsed_s':time.perf_counter()-start};(D/'CPU-contract-r123.json').write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out))
