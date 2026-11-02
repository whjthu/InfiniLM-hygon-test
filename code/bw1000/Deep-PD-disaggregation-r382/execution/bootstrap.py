"""Early sockets -> qualified loader -> same owned role loop -> original destroy."""
from pathlib import Path
import os,json,hashlib,traceback,time,socket
from contract import ROLE_NODES,check_live,load_plan,run_identity
from control_wire import EarlyRoleSockets,atom
from role_worker import RoleWorker

def execute(role,root,run,nonce,manifest_SHA,epoch,cancel_file,wire,stages,loader,leaf_factory,pin,plan):
 root=Path(root);out=root/'results'/('Deep382-'+run+'-'+role+'.json');assert not out.exists()
 q=dict(completed=False,passed=False,**run_identity(run,nonce),role=role,node=socket.gethostname(),actual_step=os.environ.get('SLURM_JOB_ID','CPU')+'.'+os.environ.get('SLURM_STEP_ID','RECORDING'),source_manifest_SHA256=manifest_SHA,
  early_sockets_before_Torch_Native_weights=False,actual_Native=False,stage_reports=[],normal_stage_loop_close=False,normal_actual8owner_destroy=False)
 save=lambda:atom(out,q);save();channel=None;holder={}
 expired=lambda:check_live(epoch,cancel_file)
 def service(ns,provider):
  leaf=leaf_factory(ns,provider,wire,pin);holder['leaf']=leaf;q['actual_Native']=leaf.actual_Native
  def stages_save(rows):q['stage_reports']=rows;save()
  try:
   result=RoleWorker(role,leaf,channel,wire,stages,expired,stages_save).run();q.update(result);save()
  finally:leaf.close_python_provider()
 def observed(ns):
  q['original_engine_destroy_completed']=ns.get('report',{}).get('engine_worker_shutdown_completed',False)
  if 'path'in ns and Path(ns['path']).exists():q['legacy_loader_receipt']=dict(path=str(ns['path']),SHA256=hashlib.sha256(Path(ns['path']).read_bytes()).hexdigest(),legacy_replica_alias=os.environ.get('DEEP_REPLICA_R9278'),actual_step=q['actual_step'],role=role)
  if 'leaf'in holder:
   leaf=holder['leaf'];owners=leaf.owner_observation.get('owners',[])
   rows=[dict(row,task_exists_after_destroy=(Path('/proc/self/task')/str(row['tid'])).exists())for row in owners]
   q['owner_exit_observation']=dict(actual_Native=leaf.actual_Native,owners=rows)
   q['normal_actual8owner_destroy']=bool(leaf.actual_Native and q['original_engine_destroy_completed']and len(rows)==8 and len({v['tid']for v in rows})==8 and not any(v['task_exists_after_destroy']for v in rows))
  save()
 try:
  expired();channel=EarlyRoleSockets(role,run,nonce,manifest_SHA,root,epoch,wire);q['early_sockets_before_Torch_Native_weights']=True;save()
  channel.authenticate_once(cancel_file);q['control_authenticated_before_weights']=True;save()
  def prepare(provider):
   from runtime_leaf import encode_once
   encode_once(provider,plan,root,run,role)
  namespace,proof,binding=loader(root=root,pin=pin,epoch=epoch,cancel_file=cancel_file,serve=service,before_weights=prepare,on_namespace_exit=observed)
  q.update(loader_proof=proof,normal_stage_loop_close=True,passed=True)
  if q['actual_Native']:assert q['normal_actual8owner_destroy']
  channel.send(dict(kind='stopped',role=role,report=q));save()
 except BaseException as e:
  q.update(error=repr(e),traceback=traceback.format_exc());save()
  if channel is not None and channel.control is not None:
   try:channel.send(dict(kind='error',role=role,error=repr(e)))
   except Exception:q['error_control_send_failed']=True
  raise
 finally:
  if channel is not None:channel.close()
  q.update(completed=True,normal_control_socket_close=True);save()
 return q

def main():
 d=Path(__file__).resolve().parent;manifest=json.loads((d/'GPU_MANIFEST.json').read_bytes());expected=os.environ['DEEP_PD_EXECUTION_SHA']
 assert hashlib.sha256((d/'GPU_MANIFEST.json').read_bytes()).hexdigest()==expected and manifest['GPU_GO']is True
 assert hashlib.sha256((d/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()==manifest['CPU_source_manifest_SHA256']
 source=json.loads((d/'SOURCE_MANIFEST.json').read_bytes());assert all(hashlib.sha256((d/n).read_bytes()).hexdigest()==h for n,h in source['files_SHA256'].items())
 from support import load
 modules=load(d);plan=load_plan(d/'root_support/BENCHMARK_PLAN.json');role=['P','D'][int(os.environ['SLURM_PROCID'])];assert socket.gethostname()==ROLE_NODES[role]
 pin=json.loads((d/'NATIVE_PIN.json').read_bytes());assert pin['ready']is True
 from qualified_loader import load as qualified
 from runtime_leaf import NamespaceLeaf
 def loader(**kwargs):return qualified(owner=Path('/refactor/incoming/Deep-paired-quality-r375/owner'),**kwargs)
 execute(role,Path('/refactor'),os.environ['DEEP_PD_RUN'],os.environ['DEEP_PD_SESSION'],expected,float(os.environ['DEEP_PD_GPU_EPOCH']),os.environ['DEEP_PD_CANCEL'],modules['state_wire'],modules['native_stages'],loader,NamespaceLeaf,pin,plan)
if __name__=='__main__':main()
