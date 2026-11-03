"""Actual mixed queue/role/splitstage/codec/socket control with recording Native leaves.

No Native, Runtime, checkpoint, Torch or GPU execution. The two fixture encodes
are a separate actual-processor consumer, never substituted by these leaves.
"""
from pathlib import Path
import tempfile,socket,threading,time,json,hashlib,copy
from unittest.mock import patch
import contract,control_wire,role_worker,bootstrap,controller,shutdown_report
from support import load
from CPU_leaves import loader_factory

D=Path(__file__).resolve().parent;modules=load(D);wire=modules['state_wire']

def port():
 s=socket.socket();s.bind(('127.0.0.1',0));n=s.getsockname()[1];s.close();return n

def actual_case(root,run,ackloss=False,fixture_inputs=None):
 plan=contract.load_plan(D/'BENCHMARK_PLAN.json')
 # Deliberately recording IDs. Actual official IDs come only from CPU_fixture.
 if fixture_inputs:contract.bind_encoders(plan,fixture_inputs)
 else:
  for spec in plan['fixtures'].values():spec['actual_prompt_token_ids_SHA256']=hashlib.sha256(json.dumps([7]*spec['actual_prompt_tokens'],separators=(',',':')).encode()).hexdigest()
 contract.ROLE_IPS.update(P='127.0.0.1',D='127.0.0.1');contract.CONTROL_PORTS.update(P=port(),D=port());p=port();control_wire.STATE_PORT=p;role_worker.STATE_PORT=p
 shared=dict(gates={i:threading.Event()for i in [5,6]},events=[],pipeline_proofs=[],quality_fail=None,fixture_inputs=fixture_inputs);holders=[];errors=[];reports=[]
 epoch=time.time();nonce='c'*64;mf='d'*64;cancel=root/'control'/('cancel-'+run)
 original_publish=shutdown_report.publish
 def recorded_shutdown(*args):
  args=list(args);q=copy.deepcopy(args[-1]);assert q['actual_Native']is False
  q.update(actual_Native=True,normal_actual8owner_destroy=True,CPU_recorded_owner_claim_not_actual=True);args[-1]=q
  return original_publish(*args)
 def role(name):
  try:reports.append(bootstrap.execute(name,root,run,nonce,mf,epoch,cancel,wire,modules['native_stages'],loader_factory(holders,name,shared),lambda ns,*args:ns['recording_leaf'],dict(ready=False,CPU_RECORDING=True),plan))
  except BaseException as e:errors.append(dict(role=name,error=repr(e)))
 original_send=wire.send_json
 def fail_ack(sock,value,deadline):
  if value.get('published')is True:raise OSError('CPU injected postcommit ACKloss')
  return original_send(sock,value,deadline)
 result=None;failure=None
 with patch.object(shutdown_report,'publish',recorded_shutdown):
  if ackloss:wire.send_json=fail_ack
  threads=[threading.Thread(target=role,args=(name,),name='CPU-role-'+name)for name in ['P','D']]
  for t in threads:t.start()
  try:result=controller.execute(root,D,run,nonce,mf,epoch,cancel,wire,modules['token_client'].TokenClient,plan)
  except BaseException as e:failure=repr(e)
  finally:
   for t in threads:t.join(10)
   wire.send_json=original_send
 assert not any(t.is_alive()for t in threads),'Owned role thread leaked'
 if ackloss:
  assert failure and errors and any(l.lib.poisoned and'poison'in l.lib.calls for l in holders)
  saved=json.loads((root/'results'/('Deep382-'+run+'-client.json')).read_bytes());assert not saved['completed']and saved['partial']['planned_external']==22
  return dict(case='actual_mixed_entry_ACKloss_failstop',passed=True,Native=False,planned_denominator=22,errors=errors)
 assert not errors and failure is None,(errors,failure)
 assert result['completed']and result['actual_stage_lifecycles']==32 and len(result['requests'])==22
 assert len(reports)==2 and all(q['normal_stage_loop_close']and q['normal_control_socket_close']and not q['actual_Native']for q in reports)
 assert sum(l.resets for l in holders)==64 and all(l.provider_closed for l in holders)
 assert sum(l.lib.calls.count('begin_export')for l in holders)==10
 assert sum(l.lib.calls.count('begin_import')for l in holders)==10
 assert len(shared['pipeline_proofs'])==2 and {p['group']for p in shared['pipeline_proofs']}=={5,6}
 for row in result['requests']:
  req=row['request'];assert req['past']in [150,8489]and req['output_budget']==(128 if req['fixture']=='L'else 1024)
  if req['mode']=='PD':
   assert row['D_report']['Native_forward_summary']['prefill_input_tokens']==0
   assert row['D_report']['Native_import_evidence']['all8_validate_then_commit']
   assert row['client_token_ID_timeline'][0]['received_monotonic']<=next(v['client_received_monotonic']for v in row['events']if v['kind']=='prefill_ready')
 for group in [4,7]:
  rows=[r['request']for r in result['requests']if r['request']['group']==group]
  for role_name in ['P','D']:assert sorted(r['fixture']for r in rows if r['co_role']==role_name)==['L','S']
 if fixture_inputs:
  assert len(shared['actual_fixture_consumed'])==22
  assert {r['fixture']for r in shared['actual_fixture_consumed']}=={'L','S'}
 return dict(case='actual_22mixed_requests32stages10tickets_loopback',passed=True,Native=False,requests=22,stages=32,resets=64,C4_active_rows_per_role=1,co_each_replica_L_and_S=True,pipeline_causal_recording=shared['pipeline_proofs'],actual_encoded_fixture_consumer=bool(fixture_inputs),fixture_consumers=shared.get('actual_fixture_consumed'))

def shutdown_case(root):
 report=dict(run='cpu-large',role='P',actual_step='CPU.RECORDING',source_manifest_SHA256='c'*64,stage_reports=[dict(recording='x'*(1100<<10))],actual_Native=True,normal_actual8owner_destroy=True,normal_stage_loop_close=True,completed=False)
 frame=shutdown_report.publish(root,'cpu-large','P',report);assert len(wire.canonical(frame))<4096
 assert frame['report_ref']['bytes']>wire.MAX_HEADER
 left,right=socket.socketpair()
 try:
  wire.send_json(left,frame,time.monotonic()+3);actual=wire.recv_json(right,time.monotonic()+3)
  assert shutdown_report.consume(root,'cpu-large','P',actual,time.monotonic()+3)==report
 finally:left.close();right.close()
 bad=copy.deepcopy(frame);bad['report_ref']['SHA256']='0'*64
 try:shutdown_report.consume(root,'cpu-large','P',bad,time.monotonic()+3)
 except AssertionError:pass
 else:raise AssertionError('Wrong immutable hash accepted')
 try:shutdown_report.publish(root,'cpu-large','P',report)
 except FileExistsError:pass
 else:raise AssertionError('Immutable snapshot overwritten')
 assert not list((root/'results').glob('*.publishing-*'))
 return dict(case='atomic_complete_shutdown_file_large_real_frame_and_negative',passed=True,Native=False,frame_bytes=len(wire.canonical(frame)),report_bytes=frame['report_ref']['bytes'],MAX_HEADER=wire.MAX_HEADER)

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--fixture-result');args=parser.parse_args();inputs=None
 if args.fixture_result:
  q=json.loads(Path(args.fixture_result).read_bytes());assert q['passed']and q['actual_prepared_encoder']and q['encodes']==2;inputs=copy.deepcopy(q['fixtures'])
  for value in inputs.values():
   raw=Path(value['private_IDs_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==value['private_IDs_file_SHA256'];value['ids']=json.loads(raw)
   assert hashlib.sha256(json.dumps(value['ids'],separators=(',',':')).encode()).hexdigest()==value['prompt_IDs_SHA256']
 with tempfile.TemporaryDirectory(prefix='pd445-')as tmp:
  root=Path(tmp);(root/'results').mkdir();(root/'control').mkdir()
  cases=[actual_case(root,'cpu-normal',fixture_inputs=inputs),actual_case(root,'cpu-ackloss',True,fixture_inputs=inputs),shutdown_case(root)]
 print(json.dumps(dict(completed=True,passed=True,GPU=0,Native_dlopen=False,weights_opened=False,Slurm=False,real_loopback_TCP=True,actual_bootstrap_role_scheduler_stages_ctypes_wire=True,recording_Native_and_provider_and_quality_only=True,cases=cases)))
if __name__=='__main__':main()
