"""Pure CPU keeper contract; fake Popen/accounting, no process/job/network."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import tempfile

from resident_p0_supervisor_r146 import ResidentP0Supervisor, NATIVE_SHA

A=Path(__file__).resolve().parent
CLIENT=A.parent/'client/client_p0_r143.py'
tests=[]
with tempfile.TemporaryDirectory(prefix='Kimi-supervisor-CPU-') as tmp:
 root=Path(tmp);(root/'results').mkdir();(root/'logs').mkdir();src=root/'source';(src/'client').mkdir(parents=True);(src/'client/client_p0_r143.py').write_bytes(CLIENT.read_bytes())
 for label in ['normal','late','wrong_step','client_failure']:
  rootcase=root/label;(rootcase/'results').mkdir(parents=True);(rootcase/'logs').mkdir();clock=[10.];calls=[];rows=[None]
  def popen(args,**kwargs):calls.append(args);return SimpleNamespace(pid=123)
  s=ResidentP0Supervisor(root=rootcase,source=src,run='CPU-test',version='CPU-'+label,env={},account=lambda name:rows[0],gpu_started=0,clock=lambda:clock[0],popen=popen)
  s.tick('910380.999');assert not calls and not s.marker.exists()
  ident={'actual_step':'910380.999','native_SHA256':NATIVE_SHA,'cards':32,'TP':16,'PP':2,'max_context_tokens':1024}
  if label=='wrong_step':ident['actual_step']='910380.998'
  s.identity.write_text(json.dumps(ident));s.ready.write_text(json.dumps({'ready':True,'service_node':'a14r2n08','actual_step':'910380.999','URL_on_service_node':'http://127.0.0.1:43844/v1'}))
  if label=='late':clock[0]=6000.
  if label=='wrong_step':
   try:s.tick('910380.999')
   except AssertionError:pass
   else:raise AssertionError('wrong step accepted')
   assert not calls and not s.marker.exists() and not s.stop.exists()
  else:
   s.tick('910380.999')
   if label=='late':assert not calls and s.stop.exists() and s.state['client_P0_passed'] is None
   else:
    assert len(calls)==1 and s.marker.exists();s.tick('910380.999');assert len(calls)==1
    assert '--gres=none' in calls[0] and calls[0][calls[0].index('-w')+1]=='a14r2n08'
    assert calls[0][calls[0].index('-t')+1]=='00:45:00'
    assert not any(k in s.marker.read_text() for k in ['argv','nonce','env'])
    s.output.mkdir();(s.output/'result.json').write_text(json.dumps({'identity_SHA256':s.state['identity_SHA256'],'completed':True,'passed':label=='normal'}))
    rows[0]=['910380.1000',s.name,'COMPLETED'if label=='normal'else'FAILED','0:0'if label=='normal'else'1:0']
    s.tick('910380.999');assert s.stop.exists() and s.state['client_P0_passed'] is (label=='normal')
    s.tick('910380.999');assert len(calls)==1
  tests.append({'case':label,'passed':True,'Native':False,'Slurm':False})
 print(json.dumps({'passed':True,'CPU_only':True,'GPU':False,'HTTP':False,'Native':False,'Slurm':False,'cases':tests}))
