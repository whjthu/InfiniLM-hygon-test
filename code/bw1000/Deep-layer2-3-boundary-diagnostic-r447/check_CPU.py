"""Necessary changed save/2pass consumer only; all Native leaves are recording.

Reuses the CPU911 high64 typed original reset/context recording leaf, not the
old 4pass/GU/shadow suite. Does not import or execute the Native/Runtime DSO.
"""
from pathlib import Path
import ctypes as C
import hashlib,json,os,sys,tempfile,time
S=Path(__file__).resolve().parent
P=S.parent/'Deep-T1-MoE-activation-quant-r438/model-diagnostic-v2'
sys.path.insert(0,str(P));sys.path.insert(0,str(S))
import numpy as np
from check_save_CPU import context
from run_teacher import run,teacher_from_provenance,ids_sha


def main():
 ref=json.loads((P/'TEACHER_REFERENCE.json').read_bytes());path=Path(os.environ['DEEP438_DIAG_TEACHER'])
 assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['SHA256']
 teacher=teacher_from_provenance(json.loads(path.read_bytes()),ref)
 prompt=list(range(8489));reference=dict(ref,prompt_IDs_SHA256=ids_sha(prompt));cases=[]
 for kind in ('normal','forward_nonzero','continuity_difference'):
  ctx,state,events,errors,callbacks,inputs=context('forward_nonzero'if kind=='forward_nonzero'else'normal')
  cap=dict(tag=None,sealed=False)
  def install(name,result,args,body):
   def safe(*values):
    try:return body(*values)
    except BaseException as error:errors.append(dict(name=name,error=repr(error)));return 9
   fn=C.CFUNCTYPE(result,*args)(safe);fn.argtypes=args;fn.restype=result;setattr(ctx.binding.lib,'deep_boundary_'+name+'_r447',fn);callbacks.append(fn)
  def arm(p,o,f):
   assert state['done']and cap['tag']is None and not cap['sealed']and p in(0,1)and o in(82,83)and f==state['f']+1==9+o
   cap['tag']=(p,o,f);return 0
  def seal():assert state['done']and cap['tag']and state['f']==cap['tag'][2];cap['sealed']=True;return 0
  def abandon():assert state['done'];cap['sealed']=True;return 0
  def clear():assert state['done'];cap.update(tag=None,sealed=False);return 0
  def count(owner):assert cap['sealed']and 0<=owner<8;return 7 if cap['tag']else 0
  def info(owner,point,out,n):
   assert cap['sealed']and n==15 and point in range(7);p,o,f=cap['tag'];residual=point in(0,1,4)
   fields=[p,o,f,owner,2 if point==0 else 3,point,1,3 if residual else 2,1,4 if residual else 4096,4096 if residual else 1,32768 if residual else 8192,0x100000000+owner,0x200000000+owner,300+owner]
   for i,v in enumerate(fields):out[i]=v
   return 0
  def copy(owner,point,out,n):
   assert state['done']and cap['sealed']and n==(32768 if point in(0,1,4)else 8192);raw=bytes(n)
   if kind=='continuity_difference'and owner==0 and point==1:raw=b'\1'+raw[1:]
   C.memmove(out,raw,n);return 0
  for name,result,args,body in [('arm',C.c_int,[C.c_int,C.c_int,C.c_uint64],arm),('seal',C.c_int,[],seal),('abandon',C.c_int,[],abandon),('clear',C.c_int,[],clear),('count',C.c_uint64,[C.c_int],count),('info',C.c_int,[C.c_int,C.c_size_t,C.POINTER(C.c_uint64),C.c_size_t],info),('copy',C.c_int,[C.c_int,C.c_size_t,C.c_void_p,C.c_size_t],copy)]:install(name,result,args,body)
  with tempfile.TemporaryDirectory()as tmp:
   out=Path(tmp)/'diagnostic';error=None
   try:q=run(ctx,prompt,teacher,out,np=np,deadline=time.time()+150,reference=reference)
   except BaseException as e:error=e;q=json.loads((out/'report.json').read_bytes())
   assert not errors,errors
   if kind=='normal':
    assert error is None and q['completed']and q['passed']and q['completed_forwards']==q['actual_forward_attempts']==184 and state['resets']==2
    assert[v[1]for v in events if isinstance(v,tuple)and v[0]=='mode']==[0,0]
    assert len(list(out.glob('pass*-boundaries/owner*.bin')))==32 and q['raw_budget_used']<=32*1024**2
    for p in out.glob('pass*-boundaries/owner*.json'):
     meta=json.loads(p.read_bytes());raw=(p.parent/meta['binary']).read_bytes();assert len(raw)==meta['bytes']==131072 and hashlib.sha256(raw).hexdigest()==meta['SHA256']
     assert len(meta['records'])==7 and meta['L2exit_equals_L3entry']
    for p in range(2):assert inputs[p*92+9:(p+1)*92]==[dict(past=8489+i,part=[v])for i,v in enumerate(teacher)]
   elif kind=='forward_nonzero':assert error and q['completed']and not q['passed']and(q['actual_forward_attempts'],q['completed_forwards'])==(12,11)and state['resets']==2
   else:assert error is None and q['early_stop']and q['completed_forwards']==91 and len(q['sequences'])==1 and state['resets']==1
   cases.append(dict(name=kind,passed=True,completed_forwards=q['completed_forwards'],attempts=q['actual_forward_attempts'],resets=state['resets'],raw_budget=q['raw_budget_used']))
 print(json.dumps(dict(completed=True,passed=True,cases=cases,actual_Native_execution=False,Runtime=False,GPU=0,SDK_execution=False,
  weights=False,actual_model_activations=False,CPU_prompt_is_shape_witness_only=True,original911_suite_replayed=False)))


if __name__=='__main__':main()
