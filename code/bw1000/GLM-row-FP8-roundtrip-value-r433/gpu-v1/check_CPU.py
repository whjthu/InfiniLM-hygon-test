"""Only changed actual schedule/raw persistence/error glue; no old suite."""
from pathlib import Path
import ast,hashlib,importlib.util,json,sys,tempfile
def run():
 d=Path(__file__).resolve().parent;g=d.parents[3];parent=g/'code/bw1000/GLM-V-on-demand-prefill-r400'
 for p in d.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
 assert(d/'checkpoint_loader.py').read_bytes()==(parent/'checkpoint_loader.py').read_bytes()
 # Same actual Native query call operands, not a replacement math leaf.
 def native_calls(p):return[ast.dump(n,include_attributes=False)for n in ast.walk(ast.parse(p.read_bytes()))if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='glm_attention_query_r398']
 assert native_calls(d/'original_consumer_r400.py')==native_calls(parent/'consumer.py')
 sys.path.insert(0,str(d));import consumer,controller
 reports=[]
 class Leaf:
  def __init__(self,root,fail):self.root=Path(root);self.fail=fail;self.closed=False;self.calls=[];self.query_CAPI_invoked=False;self.last_query_rc=None
  def check_deadline(self):pass
  def begin(self,mode):self.mode=mode;self.calls.append(('begin',mode))
  def seed(self,past,count):self.calls.append(('seed',past,count))
  def query(self,case,ids,arm):
   assert len(ids)==len(set(ids))==2048;self.query_CAPI_invoked=True;self.last_query_rc=0;self.sample=len([c for c in self.calls if c[0]=='query'])+1;self.calls.append(('query',self.mode,arm))
   if self.fail=='Native'and self.mode==1:self.last_query_rc=1;raise RuntimeError('recorded Native primary')
   return dict(recorded_Native=True)
  def persist(self,row):
   path=self.root/('sample%d.raw'%self.sample);path.write_bytes(b'recorded-only');row['raw_outputs']=[dict(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest())];row['raw_outputs_complete']=self.last_query_rc==0;self.calls.append(('persist',self.sample))
  def validate(self,row):
   assert row['raw_outputs']and Path(row['raw_outputs'][0]['path']).read_bytes()==b'recorded-only';self.calls.append(('validate',self.sample))
   if self.fail in ['numeric','double']and self.mode==1:raise RuntimeError('recorded numeric primary')
  def close(self):
   self.closed=True
   if self.fail=='double':raise RuntimeError('recorded cleanup secondary')
 for failure in [None,'numeric','Native','double']:
  with tempfile.TemporaryDirectory()as root:
   leaf=Leaf(root,failure);path=Path(root)/'report.json'
   def save(rows,error):path.write_text(json.dumps(dict(records=rows,error=error)))
   try:rows=consumer.consume(leaf,save)
   except RuntimeError as error:
    assert failure and('Native primary'in str(error)if failure=='Native'else'numeric primary'in str(error))
   else:assert failure is None and len(rows)==12 and sum(c[0]=='seed'for c in leaf.calls)==780 and sum(c[0]=='query'and c[2]for c in leaf.calls)==2
   q=json.loads(path.read_bytes());assert leaf.closed
   if failure:assert len(q['records'])==2 and q['records'][1]['raw_outputs']and q['error']
   if failure=='Native':assert q['records'][1]['Native_CAPI_returned']and not q['records'][1]['Native_CAPI_completed']and q['records'][1]['Native_CAPI_RC']==1
   if failure=='double':assert 'cleanup secondary'in q['records'][1]['close_secondary_error']and 'numeric primary'in q['error']
   reports.append(dict(case=failure or'normal',actual_new_consumer=True,real_tempfile=True,Native=False))
 cmd=controller.gpu_command('/not-created-CPU-root');assert all(x in cmd for x in ['--jobid=910374','a02r4n07','-c32','--gres=dcu:8','00:10:00'])
 source=(d/'GPU_run.py').read_text();assert'os.sched_setaffinity'not in source and 'CPU_affinity_entry=cpus'in source and '80a14a45954f4eaeed2dfca64f9be9f7ceb47d404e6e7a746a9c2062b7a285db'in source
 assert(d/'binding.py').read_bytes()==(d.parent/'diagnostic-v2/binding.py').read_bytes()and(d/'diagnostic_io.py').read_bytes()==(d.parent/'diagnostic-v2/diagnostic_io.py').read_bytes()
 return dict(passed=True,cases=reports,original_query_operands_AST_equal=True,original_loader_bytes=True,actual_command_producer=True,actual_new_typed_bindings_reused=True,old872_879_suites_rerun=False,Runtime=False,Native_dlopen=False,GPU=0,Slurm=0)
if __name__=='__main__':print(json.dumps(run(),indent=2))
