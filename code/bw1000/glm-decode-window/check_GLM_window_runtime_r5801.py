"""Small actual-Python adapter/AST closure checks only, no Torch or GPU import."""
import pathlib,json,hashlib,ast,ctypes,threading,types
r=pathlib.Path(__file__).resolve().parent;checks=0
def ok(v):
 global checks
 assert v;checks+=1
m=json.loads((r/'GLM-CROSSREQUEST-Model-source-manifest-r5799.json').read_text())
for n,h in m['source_SHA256'].items():
 ok(hashlib.sha256((r/n).read_bytes()).hexdigest()==h)
 if n.endswith('.py'):ast.parse((r/n).read_text())
for n in ['launch_GLM_crossrequest_Model_r5799.py','watch_GLM_transient_Model_terminal_r5799.py','audit_GLM_crossrequest_Model_r5799.py']:ast.parse((r/n).read_text())
tree=ast.parse((r/'glm_window_control_r5799.py').read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef));ns={'ctypes':ctypes,'json':json,'threading':threading}
class Tensor:
 def __init__(self,v):self.v=v[0]
 def item(self):return self.v
ns['torch']=types.SimpleNamespace(tensor=lambda v,**kw:Tensor(v));exec(compile(ast.Module(body=[cl],type_ignores=[]),'<actual Window>','exec'),ns)
class Fn:
 def __init__(self,f):self.f=f
 def __call__(self,*a):return self.f(*a)
for stage in [0,1]:
 log=[];sample={};report={};lib=types.SimpleNamespace(modern_glm_window_prepare_r5797=Fn(lambda ctx:log.append(('prepare',ctx))or 0),modern_glm_window_arm_r5795=Fn(lambda ctx,s,c,on:log.append(('arm',s,c,on))or 0),modern_glm_window_read_r5795=Fn(lambda ctx:json.dumps(sample).encode()))
 def control(t):t.v*=16
 w=ns['Window'](lib,42,stage,3,control,report,lambda:None);w.prepare();ok(len(log)==stage)
 for step in [62,63,64,65,66]:
  for cohort in [0,1]:
   group=list(range(cohort*8,cohort*8+8));before=len(log);w.before(step,group);expected=stage==1 and step in [63,64,65];ok(len(log)-before==int(expected))
   sample.update(layer=38,stage=1,rank=3,round=step,cohort=cohort,past=512+step-1,marker_count=32);w.after('prof-I512-D-window',step,group)
 ok(len(report.get('selected_decode_window_r5799',[]))==6*stage)
lib.modern_glm_window_arm_r5795=Fn(lambda *a:1)
try:w.before(63,list(range(8)))
except AssertionError:checks+=1
else:raise AssertionError('all16 ARM failure must stop before forward')
probe=(r/'glm_layer77_probe_r5799.py').read_text();ok(probe.count('if model_window:window.before(step,group)')==2);ok(probe.count('if model_window:window.after(label,step,group)')==2)
for body in [probe[probe.index('                def local_forward'):probe.index('                def local_forward')+12000],probe[probe.index('            for group'):]]if '            for group'in probe else [probe]:
 if 'if model_window:window.after' in body:ok(body.index('native_end=time.perf_counter()')<body.index('if model_window:window.after'))
ok("labels=['perf-I512-O-before','prof-I512-D-window','perf-I512-O-restored']"in probe)
p=r/'GLM-window-runtime-CPU-checks-r5801.json';assert not p.exists();p.write_text(json.dumps({'passed':True,'checks':checks,'scope':'Actual Window class with fake transport; source SHA/AST/pre-forward all16 failure vote and captured CAPI-end-before-getter. Not native math/stream tests.','native_CPU':'910406.182 COMPLETED0/31s, actual4TU/freshM/solemarker passed.','GPU':'NOT_RUN'},indent=2),encoding='utf8');print('PASSED',checks)
