"""Real frame/file producer-consumer regression using sealed836 scalar reports.

No Native, model, Torch, GPU or Slurm. Scheduler and already-completed Model
stages are recordings; actual private controller shutdown consumes real sockets.
"""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,sys,socket,tempfile,threading,time

D=Path(__file__).resolve().parent;BASE=D.parent
sys.path[:0]=[str(D),str(BASE)]
from shutdown_report import publish,consume
spec=importlib.util.spec_from_file_location('repair_wire',BASE/'root_support/state_wire.py');wire=importlib.util.module_from_spec(spec);spec.loader.exec_module(wire)
spec=importlib.util.spec_from_file_location('repair_controller',D/'controller.py');controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)

def main():
 a=argparse.ArgumentParser();a.add_argument('--sealed-dir',required=True);a.add_argument('--output',required=True);a=a.parse_args()
 source=Path(a.sealed_dir);run='pd-20261102-a';checks=[];refs=[]
 def read(name):
  p=source/name;b=p.read_bytes();refs.append(dict(path=str(p),SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b)));return json.loads(b)
 reports={r:read('Deep382-'+run+'-'+r+'.json')for r in ['P','D']}
 original=read('Deep382-'+run+'-client.json');recording=original['partial']
 normalized={}
 for role,v in reports.items():
  q=copy.deepcopy(v)
  for k in ['error','traceback','normal_control_socket_close']:q.pop(k,None)
  q['completed']=False;normalized[role]=q
  n=len(wire.canonical(dict(kind='stopped',role=role,report=q)))
  assert n>wire.MAX_HEADER
  try:wire.send_json(None,dict(kind='stopped',role=role,report=q),time.monotonic()+1)
  except ValueError as e:assert str(e)=='Header exceeds bound'
  else:raise AssertionError('Original oversize frame unexpectedly accepted')
  checks.append(dict(case='original_'+role+'_actual_report_exceeds_unchanged_bound',passed=True,wire_bytes=n,MAX_HEADER=wire.MAX_HEADER))
 with tempfile.TemporaryDirectory(prefix='pd382-shutdown-')as tmp:
  root=Path(tmp);(root/'results').mkdir();frames={};frozen={}
  for role,q in normalized.items():
   frame=publish(root,run,role,q);frames[role]=frame
   p=root/'results'/('Deep382-'+run+'-'+role+'-shutdown.json');frozen[role]=p.read_bytes()
   assert len(wire.canonical(frame))<4096
   left,right=socket.socketpair()
   try:
    wire.send_json(left,frame,time.monotonic()+2);received=wire.recv_json(right,time.monotonic()+2);assert consume(root,run,role,received,time.monotonic()+2)==q
   finally:left.close();right.close()
   checks.append(dict(case='actual_'+role+'_publish_wire_consume_large_report',passed=True,frame_bytes=len(wire.canonical(frame)),report_bytes=len(frozen[role])))
  for label,change in [('wrong_path',lambda f:f['report_ref'].update(path='/refactor/results/other.json')),('wrong_SHA',lambda f:f['report_ref'].update(SHA256='0'*64)),('wrong_bytes',lambda f:f['report_ref'].update(bytes=f['report_ref']['bytes']+1)),('wrong_count',lambda f:f['summary'].update(stage_report_count=0)),('wrong_role',lambda f:f.update(role='D')),('wrong_owner',lambda f:f['summary'].update(normal_actual8owner_destroy=False))]:
   f=copy.deepcopy(frames['P']);change(f)
   try:consume(root,run,'P',f,time.monotonic()+.05)
   except (AssertionError,TimeoutError):pass
   else:raise AssertionError(label)
   checks.append(dict(case=label,passed=True))
  try:publish(root,run,'P',normalized['P'])
  except FileExistsError:checks.append(dict(case='exclusive_snapshot_no_overwrite',passed=True))
  else:raise AssertionError('Snapshot overwritten')
  # The subsequent mutable main report cannot change the referenced snapshot.
  (root/'results'/('Deep382-'+run+'-P.json')).write_text('{"completed":true}')
  assert consume(root,run,'P',frames['P'],time.monotonic()+2)==normalized['P']
  checks.append(dict(case='main_report_finally_rewrite_keeps_reference_stable',passed=True))
  p=root/'results'/('Deep382-'+run+'-P-shutdown.json');p.unlink()
  def delayed_visibility():
   time.sleep(.04)
   with p.open('xb')as f:
    f.write(frozen['P'][:64]);f.flush();time.sleep(.04);f.write(frozen['P'][64:])
  writer=threading.Thread(target=delayed_visibility);writer.start();tick=time.monotonic()
  assert consume(root,run,'P',frames['P'],tick+1)==normalized['P'];writer.join(1)
  assert not writer.is_alive()and time.monotonic()-tick>=.06
  checks.append(dict(case='passive_missing_then_partial_NFS_visibility',passed=True,wait_s=time.monotonic()-tick))
  p.unlink();tick=time.monotonic()
  try:consume(root,run,'P',frames['P'],tick+.05)
  except TimeoutError:pass
  else:raise AssertionError('Missing immutable report exceeded original deadline')
  assert time.monotonic()-tick<.15
  checks.append(dict(case='visibility_capped_by_original_cleanup_deadline',passed=True))
  p.write_bytes(frozen['P']);bad=copy.deepcopy(frames['P']);bad['report_ref']['SHA256']='f'*64;tick=time.monotonic()
  try:consume(root,run,'P',bad,tick+10)
  except AssertionError:pass
  else:raise AssertionError('Complete wrong hash accepted')
  assert time.monotonic()-tick<.1
  checks.append(dict(case='complete_wrong_hash_fails_without_visibility_wait',passed=True))
  for role in ['P','D']:
   p=root/'results'/('Deep382-'+run+'-'+role+'-shutdown.json');assert p.read_bytes()==frozen[role];p.unlink()
  class RecordedScheduler:
   def __init__(self,*args,**kwargs):pass
   def run(self):return copy.deepcopy(recording)
   def failure(self,*args):raise AssertionError(args)
  controller.Scheduler=RecordedScheduler
  peers={};actors=[];errors=[]
  def actor(role,peer):
   try:
    wire.send_json(peer,dict(kind='weights_ready',role=role,receipt=original['role_ready_receipts'][role]),time.monotonic()+10)
    assert wire.recv_json(peer,time.monotonic()+10)==dict(kind='stop')
    wire.send_json(peer,publish(root,run,role,normalized[role]),time.monotonic()+10)
   except BaseException as e:errors.append(repr(e))
   finally:peer.close()
  for role in ['P','D']:
   peers[role],remote=socket.socketpair();t=threading.Thread(target=actor,args=(role,remote));actors.append(t);t.start()
  plan=json.loads((BASE/'root_support/BENCHMARK_PLAN.json').read_bytes());saved=[]
  result=controller.run_connected(peers,wire,None,plan,time.time(),root/'cancel-does-not-exist',saved.append,stopped_report_loader=lambda role,frame:consume(root,run,role,frame,time.monotonic()+10))
  for t in actors:t.join(10)
  assert not errors and not any(t.is_alive()for t in actors)
  assert result['actual_stage_lifecycles']==22 and result['all16_actual_owners_destroyed']and result['quality_all_task_and_natural_EOS']and result['completed']
  checks.append(dict(case='actual_private_controller_shutdown_two_role_socket_receipts',passed=True,recorded_requests=15,recorded_stages=22,recorded_owner_conclusion=16))
 result=dict(completed=True,passed=True,GPU=0,Slurm_jobs=0,Native_dlopen=False,Model_loads=0,MAX_HEADER_unchanged=wire.MAX_HEADER,checks=checks,checks_count=len(checks),sealed_input_refs=refs,scope='Actual immutable file producer + unchanged real frame codec + actual private controller shutdown. Recorded836 stage/quality/owner metadata; not new Native/Model/owner execution.',original_failure_retained=True)
 Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
