"""Changed teacher/typed-storage consumer. All Native math and GPU leaves recorded."""
from pathlib import Path
import ast,ctypes as C,hashlib,json,os,struct,tempfile,threading,time,types
import numpy as np
from capture_api import ROLE_FIELDS,Capture,Budget
from teacher_runner import run,plan,ids_sha,teacher_from_provenance
S=Path(__file__).resolve().parent
HIGH=0x123456789abcdef0

def context(kind):
 parent=S.parent/'model-v1';p=parent/'original_native_context.py';proof=json.loads((parent/'ACTUAL_MODEL_REACHABILITY.json').read_bytes())
 assert hashlib.sha256(p.read_bytes()).hexdigest()==proof['qualified_native_context_SHA256']
 tree=ast.parse(p.read_bytes());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='NativeContext');methods=[n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name in['reset','check_thread']]
 env={'threading':threading};exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),'actual-qualified-context-methods','exec'),env)
 state=dict(f=0,mode=0,done=True,armed=False,resets=0,tag=None,sealed=False);callbacks=[];errors=[];events=[];lib=types.SimpleNamespace()
 def install(name,result,args,body):
  def safe(*values):
   try:return body(*values)
   except BaseException as e:errors.append(dict(name=name,error=repr(e)));return 9
  fn=C.CFUNCTYPE(result,*args)(safe);fn.argtypes=args;fn.restype=result;setattr(lib,name,fn);callbacks.append(fn)
 def reset(ctx):
  assert ctx==HIGH;state.update(f=0,done=True);state['resets']+=1;events.append('real-typed-reset-recorded');return 0
 def forward(ctx,n,rows,*unused):
  assert ctx==HIGH and state['done']and state['armed'];state['done']=False;state['f']+=1
  if kind=='forward_nonzero'and state['f']==12:state['done']=True;return 7
  state['done']=True;return 0
 def clear():assert state['done'];state.update(mode=0,armed=False,f=0);return 0
 def setmode(mode):assert state['done']and mode in[0,1];state.update(mode=mode,armed=True);events.append(('mode',mode));return 0
 def read(owner,out,n):
  assert state['done']and 0<=owner<8 and n==12;f=state['f'];t1=43*max(0,f-9);mode=state['mode'];values=[43*f,t1,mode*t1,t1,mode*t1,(1-mode)*t1,t1,mode*t1,(1-mode)*t1,43*f,0,mode]
  for k,v in enumerate(values):out[k]=v
  return 0
 install('modern_native_engine_reset',C.c_int,[C.c_void_p],reset);install('modern_native_engine_destroy',C.c_int,[C.c_void_p],lambda ctx:0)
 install('modern_native_engine_forward',C.c_int,[C.c_void_p,C.c_size_t,C.c_size_t]+[C.c_void_p]*8,forward)
 install('deep_model_actquant_clear_r438',C.c_int,[],clear);install('deep_model_actquant_set_r438',C.c_int,[C.c_int],setmode)
 install('deep_model_actquant_read_r438',C.c_int,[C.c_int,C.POINTER(C.c_uint64),C.c_size_t],read);install('deep_model_actquant_metadata_r438',C.c_char_p,[],lambda:None)
 def arm(p,o,f):
  assert state['done']and not state['sealed']and state['tag']is None and f==state['f']+1 and f==9+o and o in[82,83]and p in range(4);state['tag']=(p,o,f);return 0
 def seal():assert state['done']and state['tag']and state['f']==state['tag'][2];state['sealed']=True;return 0
 def capclear():assert state['done'];state.update(tag=None,sealed=False);return 0
 def abandon():assert state['done'];state['sealed']=True;return 0
 def count(owner):assert state['sealed']and 0<=owner<8;return 86 if state['tag']is not None else 0
 def field(role,index):
  name=ROLE_FIELDS[role][index];rows=6 if role==0 else 1
  code,shape=(1,(1,4096))if name=='hidden'else(4,(1,6))if name=='route_ids'else(3,(1,6))if name=='route_weights'else(1,(rows,512))if name.endswith('GU')else(1,(rows,256))if name.endswith('activation')else(2,(rows,256))if name.endswith('I8')else(3,(rows,1))
  width={1:2,2:1,3:4,4:8}[code];n=shape[0]*shape[1]*width
  raw=struct.pack('<6q',*range(6))if name=='route_ids'else struct.pack('<6f',*[.25]*6)if name=='route_weights'else struct.pack('<f',1e-8)*rows if name.endswith('scale')else bytes(n)
  return name,[code,*shape,width,n],raw
 def info(owner,index,out,n):
  assert state['sealed']and n==12;layer,role=divmod(index,2);p,o,f=state['tag'];fields=len(ROLE_FIELDS[role]);total=sum(len(field(role,k)[2])for k in range(fields));values=[p,o,f,layer,owner,role,fields,total,0x100000000+owner,0x200000000+owner,300+owner,0x400000000+index]
  for k,v in enumerate(values):out[k]=v
  return 0
 def tensor(owner,index,slot,out,n):
  assert state['sealed']and n==5;_,values,raw=field(index%2,slot)
  if kind=='bad_dtype'and owner==0 and index==0 and slot==0:values[0]=3
  for k,v in enumerate(values):out[k]=v
  return 0
 def copy(owner,index,slot,out,n):
  assert state['sealed']and out;name,values,raw=field(index%2,slot);assert n==len(raw)
  if kind=='shadow_diff'and owner==0 and index==0 and name=='candidate_I8':raw=b'\1'+raw[1:]
  C.memmove(out,raw,n);return 0
 for name,result,args,fn in [('arm',C.c_int,[C.c_int,C.c_int,C.c_uint64],arm),('seal',C.c_int,[],seal),('clear',C.c_int,[],capclear),('abandon',C.c_int,[],abandon),('count',C.c_uint64,[C.c_int],count),('record',C.c_int,[C.c_int,C.c_size_t,C.POINTER(C.c_uint64),C.c_size_t],info),('tensor',C.c_int,[C.c_int,C.c_size_t,C.c_size_t,C.POINTER(C.c_uint64),C.c_size_t],tensor),('copy',C.c_int,[C.c_int,C.c_size_t,C.c_size_t,C.c_void_p,C.c_size_t],copy)]:install('deep_model_capture_'+name+'_r438v2',result,args,fn)
 def check(code):
  if code:raise RuntimeError('Recorded original Native rc='+str(code))
 ctx=types.SimpleNamespace(ctx=HIGH,thread=threading.get_ident(),closed=False,cfg={'vocab_size':129280},binding=types.SimpleNamespace(lib=lib,check=check))
 ctx.check_thread=types.MethodType(env['check_thread'],ctx);ctx.reset=types.MethodType(env['reset'],ctx);input_rows=[]
 def original(part,positions,requests,pasts,totals,offsets):
  assert len(part)==len(positions)and requests==[0]and positions==list(range(pasts[0],totals[0]))and offsets==[0,len(part)]and totals[0]==pasts[0]+len(part)
  input_rows.append(dict(past=pasts[0],part=list(part)));code=lib.modern_native_engine_forward(HIGH,len(part),1,*([None]*8))
  logits=np.zeros((1,129280),dtype=np.uint16);logits[0,30722]=0x4200
  return code,logits,np.array([30722],dtype=np.int64)
 ctx.forward=original;return ctx,state,events,errors,callbacks,input_rows

def main():
 card=json.loads((S/'TEACHER_REFERENCE.json').read_bytes());private=Path(os.environ['DEEP438_DIAG_TEACHER']);assert hashlib.sha256(private.read_bytes()).hexdigest()==card['SHA256'];teacher=teacher_from_provenance(json.loads(private.read_bytes()),card)
 # CPU uses positional shape witnesses only. These are not actual Model IDs
 # or activations. Production run accepts only real prepared IDs matching the
 # frozen prompt hash; no Native/runtime/weights are executed in this job.
 prompt=list(range(8489));reference=dict(card,prompt_IDs_SHA256=ids_sha(prompt));cases=[]
 for kind in ['normal','forward_nonzero','shadow_diff','bad_dtype']:
  ctx,state,events,errors,callbacks,inputs=context(kind)
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp)/'diagnostic';error=None
   try:result=run(ctx,prompt,teacher,root,np=np,deadline=time.time()+150,reference=reference)
   except BaseException as e:error=e;result=json.loads((root/'report.json').read_bytes())
   assert not errors,errors
   if kind=='normal':
    assert error is None and result['completed']and result['passed']and result['completed_forwards']==368 and state['resets']==4 and [v[1]for v in events if isinstance(v,tuple)and v[0]=='mode']==[0,1,0,1]
    assert len(list(root.glob('pass*-owners/owner*.bin')))==64 and result['raw_budget_used']<=128*1024**2
    for p in root.glob('pass*-owners/owner*.json'):
     metadata=json.loads(p.read_bytes());raw=(p.parent/metadata['binary']).read_bytes();assert len(raw)==metadata['bytes']and hashlib.sha256(raw).hexdigest()==metadata['SHA256']
     for record in metadata['records']:
      for field in record['tensors'].values():
       segment=raw[field['offset']:field['offset']+field['bytes']];assert len(segment)==field['bytes']and hashlib.sha256(segment).hexdigest()==field['SHA256']
    for seq in range(4):
     rows=inputs[seq*92:(seq+1)*92];assert rows[9:]==[dict(past=8489+i,part=[v])for i,v in enumerate(teacher)]
   elif kind=='forward_nonzero':assert error and result['actual_forward_attempts']==12 and result['completed_forwards']==11 and state['resets']==2
   elif kind=='shadow_diff':assert error is None and result['early_stop']and result['completed_forwards']==91 and len(result['sequences'])==1 and state['resets']==1
   else:
    assert error and 'dtype'in repr(error)and result['completed_forwards']==91
    partial=json.loads((root/'partial-on-failure/owner0.json').read_bytes());assert partial['partial']and not partial['qualified']and partial['records'][0]['fields'][0]['info'][0]==3 and partial['records'][0]['fields'][0]['info'][3]==2
   cases.append(dict(name=kind,passed=True,actual_forward_attempts=result['actual_forward_attempts'],completed_forwards=result['completed_forwards'],resets=state['resets'],recording_native_error=repr(error)if error else None,raw_budget_used=result.get('raw_budget_used',0)))
 budget=Budget(3);budget.charge(3)
 try:budget.charge(1)
 except RuntimeError:pass
 else:raise AssertionError('Budget not bounded')
 cap=Capture(context('normal')[0].binding.lib,Budget());guard=types.SimpleNamespace(active=True,quiescent=False,poisoned=False)
 try:cap.arm(guard,0,82,91)
 except RuntimeError:pass
 else:raise AssertionError('HostReturned-only capture arm accepted')
 q=dict(completed=True,passed=True,cases=cases,typed_saved_bytes_and_high64_ctx=True,actual_qualified_reset_thread_methods=True,original902_904_suites_replayed=False,CPU_shape_witness_prompt_not_Model_input=True,Native_and_HTTP_and_GPU_leaves_recording_only=True,actual_Native_execution=False,Runtime=False,GPU=0,weights=False,actual_Model_activation_captured=False)
 print(json.dumps(q))
if __name__=='__main__':main()
