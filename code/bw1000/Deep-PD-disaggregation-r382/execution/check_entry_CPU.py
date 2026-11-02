"""Actual bootstrap/control/scheduler/role/codec/wire/client with recording Native leaves."""
from pathlib import Path
import tempfile,socket,threading,time,json,hashlib,sys,ast,traceback
from unittest.mock import patch
import contract,control_wire,role_worker,bootstrap,controller
from support import load
from CPU_leaves import loader_factory,Leaf

D=Path(__file__).resolve().parent;modules=load(D);wire=modules['state_wire'];plan=contract.load_plan(D/'root_support/BENCHMARK_PLAN.json')
def port():
 s=socket.socket();s.bind(('127.0.0.1',0));value=s.getsockname()[1];s.close();return value
def actual_case(root,run,ackloss=False,quality_fail=None):
 contract.ROLE_IPS.update(P='127.0.0.1',D='127.0.0.1');contract.CONTROL_PORTS.update(P=port(),D=port());stateport=port();control_wire.STATE_PORT=stateport;role_worker.STATE_PORT=stateport
 shared=dict(gates={i:threading.Event()for i in [7,8]},events=[],pipeline_proofs=[],quality_fail=quality_fail);holders=[];errors=[];reports=[]
 epoch=time.time();nonce='c'*64;mf='d'*64;cancel=root/'control'/('cancel-'+run)
 def role(name):
  try:reports.append(bootstrap.execute(name,root,run,nonce,mf,epoch,cancel,wire,modules['native_stages'],loader_factory(holders,name,shared),lambda ns,*args:ns['recording_leaf'],dict(ready=False,CPU_RECORDING=True),plan))
  except BaseException as e:errors.append(dict(role=name,error=repr(e)))
 original=wire.send_json
 def fail_ack(sock,value,deadline):
  if value.get('published')is True:raise OSError('CPU injected ACKloss after Nativecommit recording')
  return original(sock,value,deadline)
 if ackloss:wire.send_json=fail_ack
 threads=[threading.Thread(target=role,args=(name,),name='CPU-bootstrap-'+name,daemon=False)for name in ['P','D']]
 for t in threads:t.start()
 result=None;failure=None
 try:result=controller.execute(root,D,run,nonce,mf,epoch,cancel,wire,modules['token_client'].TokenClient,plan)
 except BaseException as e:failure=repr(e)
 finally:
  for t in threads:t.join(8)
  wire.send_json=original
 assert not any(t.is_alive()for t in threads),'Owned CPU bootstrap thread remained alive'
 if ackloss:
  assert failure and errors and any(l.lib.poisoned and'poison'in l.lib.calls for l in holders)
  saved=json.loads((root/'results'/('Deep382-'+run+'-client.json')).read_bytes());assert not saved['completed']and saved['partial']['planned_external']==15
  assert all(json.loads((root/'results'/('Deep382-'+run+'-'+r+'.json')).read_bytes())['normal_control_socket_close']for r in ['P','D'])
  return dict(kind='actual_wholeentry_postcommit_ACKloss',passed=True,sticky_generation_poison_recorded=True,planned_denominator=15,partial_first_token_retained=True,errors=errors,no_actual_Native=True)
 if quality_fail:
  assert not errors and failure is None and not result['completed']and result['warm_gate_failed']and result['execution_shutdown_completed']
  assert result['actual_stage_lifecycles']==4 and sum(r['admitted']for r in result['requests'])==3 and len(result['requests'])==15
  assert len(reports)==2 and all(q['normal_control_socket_close']and q['normal_stage_loop_close']for q in reports)
  return dict(kind=quality_fail+'_stops_formal_normal_destroy',passed=True,actual_requests=3,planned_denominator=15,formal_not_executed=12,stage_lifecycles=4,Native=False)
 assert not errors and failure is None and result['completed']and result['actual_stage_lifecycles']==22
 assert len(result['requests'])==15 and not result['all16_actual_owners_destroyed']
 assert len(shared['pipeline_proofs'])==2 and {p['group']for p in shared['pipeline_proofs']}=={7,8}
 assert sum(len(q['stage_reports'])for q in reports)==22 and all(q['normal_control_socket_close']for q in reports)
 assert sum(l.resets for l in holders)==44 and all(l.provider_closed for l in holders)
 assert sum(l.lib.calls.count('begin_export')for l in holders)==7 and sum(l.lib.calls.count('begin_import')for l in holders)==7
 for row in result['requests']:
  if row['request']['mode']=='PD':
   ready=next(e['client_received_monotonic']for e in row['events']if e['kind']=='prefill_ready')
   assert row['client_token_ID_timeline'][0]['received_monotonic']<=ready
   assert row['D_report']['Native_forward_summary']['prefill_input_tokens']==0
 # The explicitly CPU-recorded shared clock proves source release/reset precedes
 # P2prefill_enter while D1's finaltoken is causally blocked on that entry.
 for group in [7,8]:
  reqs=[r['request']for r in result['requests']if r['request']['group']==group];one,two=reqs
  release=next(e['tick']for e in shared['events']if e['role']=='P'and e['event']=='export_snapshot_release'and e['epoch']==one['epoch'])
  reset=next(e['tick']for e in shared['events']if e['role']=='P'and e['event']=='reset'and e['epoch']==one['epoch']and e['tick']>=release)
  entered=next(e['tick']for e in shared['events']if e['role']=='P'and e['event']=='stage_enter'and e['epoch']==two['epoch'])
  assert release<=reset<=entered
 return dict(kind='actual_wholeentry15requests22stages',passed=True,requests=15,co=8,PD=7,stage_lifecycles=22,reset_calls=44,export_import7=True,recorded_provider_parameterload_and_owners_only=True,real_bootstrap_control_socket_splitstages_ctypes_wire_tokenclient=True,physical_gpu_overlap='NOT_TESTED',pipeline_causal_CPU=shared['pipeline_proofs'])

def main():
 cases=[]
 owner=Path('/refactor/incoming/Deep-paired-quality-r375/owner')
 if not owner.exists():owner=D.parents[3]/'code/bw1000/Deep-paired-quality-r375/repair-v4/owner'
 sys.path.insert(0,str(owner));import bootstrap_r206b as parent
 from qualified_loader import transform
 raw=parent.BASE
 if not raw.exists():raw=D.parents[3]/'code/bw1000/resume-20261026/Deep-C4-independent64-private-r9278/modern_Deep_independent64_r9278.py'
 _,proof=transform(raw.read_bytes(),parent.transform,52);assert proof['Native_forward_AST_unchanged']and proof['PD_private_patch']==dict(link_compare=1,linked_report=1,create_deadline=1)
 cases.append(dict(kind='actual_BASE_and375_transform_private52_AST_keywordcounts',passed=True,Native=False))
 with tempfile.TemporaryDirectory(prefix='pd382-entry-')as tmp:
  root=Path(tmp);(root/'results').mkdir();(root/'control').mkdir()
  cases.append(actual_case(root,'cpu-normal'))
  cases.append(actual_case(root,'cpu-ackloss',True))
  cases.append(actual_case(root,'cpu-warm-wrong',quality_fail='warm_wrong'))
  cases.append(actual_case(root,'cpu-warm-cap',quality_fail='warm_cap'))
  called=[]
  for kind in ['expired','canceled']:
   cancel=root/'control'/('cancel-'+kind)
   if kind=='canceled':cancel.write_text('{}')
   try:bootstrap.execute('P',root,'cpu-'+kind,'c'*64,'d'*64,time.time()-3541 if kind=='expired'else time.time(),cancel,wire,modules['native_stages'],lambda **kw:called.append(kw),lambda *args:None,{},plan)
   except (TimeoutError,RuntimeError):pass
   else:raise AssertionError('Original deadline/cancel not enforced')
   assert not called;cases.append(dict(kind=kind+'_before_socket_provider_create_load',passed=True))
  shared=dict(events=[],gates={},pipeline_proofs=[]);leaf=Leaf(wire,'D',shared);gen=leaf.lib.generation
  assert leaf.lib.deep_pd_state_poison_r382(1,gen-1)==1 and not leaf.lib.poisoned
  assert leaf.lib.deep_pd_state_poison_r382(1,gen)==0
  try:leaf.reset()
  except RuntimeError:pass
  else:raise AssertionError('Sticky poison reset revived context')
  cases.append(dict(kind='ctypes_expected_generation_poison_stale_reject_no_reset_revive',passed=True))
  import launch_once,keeper
  class RecordingProcess:
   pid=-382
  popens=[]
  def popen(*args,**kw):popens.append(args);return RecordingProcess()
  (root/'logs').mkdir()
  with patch.object(launch_once,'R',root),patch.object(launch_once,'verify',lambda *args:None),patch.object(launch_once.subprocess,'Popen',popen):
   first=launch_once.launch(D,'a'*64,'b'*40,'cpu-one-intent');assert first['CPU_keeper_submitted']and not first['GPU_submitted_by_invoker']
   try:launch_once.launch(D,'a'*64,'b'*40,'cpu-one-intent')
   except FileExistsError:pass
   else:raise AssertionError('Immutable intent replay accepted')
   assert len(popens)==1
  with patch.object(launch_once.subprocess,'Popen',popen):
   try:launch_once.launch(root/'no-GPU-manifest','a'*64,'b'*40,'cpu-no-permission')
   except FileNotFoundError:pass
   else:raise AssertionError('Missing GPUmanifest submitted')
   assert len(popens)==1
  argv=keeper.gpu_arguments(root,D,'CPU-RECORDING');assert all(v in argv for v in ['-N2','-n2','--ntasks-per-node=1','-c32','--gres=dcu:8','--mem=240000M','01:00:00'])and '-n16'not in argv
  cases.append(dict(kind='actual_once_intent_duplicate_and_missingGPUmanifest0spawn_plus16cardSlurmargv',passed=True,Slurm_Popen_recorded_only=True,Native=False))
 print(json.dumps(dict(completed=True,passed=True,GPU=0,Native_dlopen=False,checkpoint_opened=False,official_encoding_executed=False,actual_Slurm=False,real_loopback_TCP=True,root_dependencies=json.loads((D/'DEPENDENCIES.json').read_bytes()),cases=cases)))
if __name__=='__main__':main()
