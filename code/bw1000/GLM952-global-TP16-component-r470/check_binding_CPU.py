"""Actual ctypes caller through real libffi; Native functions are recording leaves."""
import ctypes,runpy
from pathlib import Path
def run():
 api=runpy.run_path(str(Path(__file__).with_name('binding.py')));ctx=0x1234567800000042;records=[];funcs=[]
 class Leaf:pass
 lib=Leaf();P=ctypes.c_void_p;N=ctypes.c_size_t;I=ctypes.c_int;C=ctypes.c_char_p
 specs={'error':([],C),'id_size':([],N),'id':([P,N],I),'create':([I,I,N,N,C,P,P,N],P),'geometry':([P],C),'probe':([P],I),'close':([P],I)}
 for name,(args,ret)in specs.items():
  def callback(*a,_name=name):
   if _name=='id_size':return 32
   if _name=='create':records.append(dict(node=a[0],device=a[1],rows=a[2],capacity=a[3],cfg=a[4].decode(),world=ctypes.string_at(a[5],a[7]).hex(),tp=ctypes.string_at(a[6],a[7]).hex()));return ctx
   if _name in ['probe','close']:records.append(dict(op=_name,ctx=a[0]));return 0
   return b'{}'if _name=='geometry'else b'recording'if _name=='error'else 0
  f=ctypes.CFUNCTYPE(ret,*args)(callback);funcs.append(f);setattr(lib,'glm_global_tp16_'+name+'_r470',f)
 api['declare'](lib);value=api['create'](lib,1,7,{'glm_global_tp16_r470':True},b'w'*32,b't'*32);assert value==ctx
 assert lib.glm_global_tp16_probe_r470(ctx)==lib.glm_global_tp16_close_r470(ctx)==0
 assert records[0]['rows']==1 and records[0]['capacity']==32768 and records[0]['world']=='77'*32 and records[0]['tp']=='74'*32 and records[-1]['ctx']==ctx
 bad=0
 for node,device,w,t in [(2,0,b'w'*32,b't'*32),(0,8,b'w'*32,b't'*32),(0,0,b'w'*31,b't'*32),(0,0,b'w'*32,b'w'*32)]:
  try:api['create'](lib,node,device,{'glm_global_tp16_r470':True},w,t)
  except AssertionError:bad+=1
 assert bad==4 and len(records)==3
 return dict(passed=True,actual_high64_ctx=ctx,negative_cases=bad,Native_leaf='recording',Native_dlopen=False,Runtime=False,GPU=False)
if __name__=='__main__':
 import json;print(json.dumps(run()))
