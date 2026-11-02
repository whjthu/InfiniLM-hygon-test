"""One-clock tokenID client and C1 P/D schedule over two authenticated role sockets."""
from pathlib import Path
import time,threading,traceback,json,secrets,os,hashlib
from scheduler import Scheduler
from contract import check_live,load_plan,public_request
from control_wire import connect_role_once,atom

def run_connected(sockets,wire,TokenClient,plan,epoch,cancel_file,save,*,client_clock=time.monotonic,stopped_report_loader=None):
 receipts={};hard=lambda:time.monotonic()+max(.001,epoch+3600-time.time())
 for role in ['P','D']:
  event=wire.recv_json(sockets[role],hard());assert event['kind']=='weights_ready'and event['role']==role
  receipts[role]=event['receipt']
 def send(role,value):wire.send_json(sockets[role],value,hard())
 scheduler=Scheduler(plan,send,TokenClient,lambda:secrets.token_hex(32),clock=client_clock,check=lambda:check_live(epoch,cancel_file),save=lambda value:save(dict(value,role_ready_receipts=receipts)),client_deadline=client_clock()+max(.001,epoch+3540-time.time()))
 stopped={};lock=threading.Lock();errors=[]
 def reader(role):
  try:
   while True:
    frame=wire.recv_json(sockets[role],hard())
    if frame['kind']=='stopped':
     assert frame['role']==role
     assert stopped_report_loader is not None
     report=stopped_report_loader(role,frame)
     with lock:stopped[role]=report
     return
    scheduler.receive(role,frame)
  except BaseException as e:
   with lock:errors.append(dict(role=role,error=repr(e),traceback=traceback.format_exc()))
   scheduler.failure(role,e)
 threads=[threading.Thread(target=reader,args=(role,),name='PD-client-'+role,daemon=False)for role in ['P','D']]
 for t in threads:t.start()
 try:
  result=scheduler.run()
  for role in ['P','D']:send(role,dict(kind='stop'))
  while True:
   with lock:
    if errors:raise RuntimeError(errors)
    if len(stopped)==2:break
   assert time.time()<epoch+3600,'Original GPU cleanup deadline exhausted';time.sleep(.02)
  actual_stages=sum(len(v['stage_reports'])for v in stopped.values());skips=sum(bool(r.get('D_skipped_early_P_EOS'))for r in result['requests'])
  expected=sum(1 if r['request']['mode']=='co'else 1 if r.get('D_skipped_early_P_EOS')else 2 for r in result['requests']if r.get('client',{}).get('completed'))
  assert actual_stages==expected
  if not result.get('warm_gate_failed'):assert actual_stages==plan['planned_role_request_lifecycles']['total_stage_lifecycles']-skips
  quality=[]
  for row in result['requests']:
   if not row['admitted']:
    quality.append(dict(index=row['request']['index'],task_correct=None,natural_EOS=None,not_executed_after_warm_failure=True));continue
   req=row['request'];source=row[req['co_role']+'_report']if req['mode']=='co'else row['P_report']if row.get('D_skipped_early_P_EOS')else row['D_report']
   assert source['generated_token_ids']==[e['token_id']for e in row['client_token_ID_timeline']]
   if req['mode']=='PD'and not row.get('D_skipped_early_P_EOS'):assert row['D_report']['Native_forward_summary']['prefill_input_tokens']==0
   quality.append(dict(index=req['index'],task_correct=source['quality']['task_correct'],natural_EOS=source['natural_EOS'],finish_reason=source['finish_reason']))
  result.update(role_ready_receipts=receipts,role_stopped_receipts=stopped,completed=not result.get('warm_gate_failed',False),execution_shutdown_completed=True,actual_stage_lifecycles=actual_stages,early_P_EOS_skipped_D=skips,quality_summary=quality,quality_all_task_and_natural_EOS=all(r['task_correct']and r['natural_EOS']for r in quality),clock_scope='One actual controller monotonic tokenID delivery clock; no HTTP/SSE/publicSLO qualification',all16_actual_owners_destroyed=all(v.get('normal_actual8owner_destroy')for v in stopped.values()))
  save(result);return result
 except BaseException:
  save(dict(completed=False,error=traceback.format_exc(),partial=scheduler.snapshot(),role_ready_receipts=receipts,reader_errors=errors))
  raise
 finally:
  for s in sockets.values():
   try:s.shutdown(2)
   except OSError:pass
   s.close()
  for t in threads:t.join(timeout=5)
  if any(t.is_alive()for t in threads):raise RuntimeError('Owned tokenID reader thread failed to close')

def execute(root,directory,run,nonce,manifest_SHA,epoch,cancel_file,wire,TokenClient,plan,progress=lambda value:None):
 root=Path(root);result=root/'results'/('Deep382-'+run+'-client.json');assert not result.exists();sockets={}
 def save(value):
  value=dict(value,run=run,source_manifest_SHA256=manifest_SHA,session_nonce_SHA256=hashlib.sha256(nonce.encode()).hexdigest(),all_service_cards=16,public_HTTP_SLO=False)
  atom(result,value)
  progress(value)
 try:
  for role in ['P','D']:sockets[role]=connect_role_once(role,run,nonce,manifest_SHA,root,epoch,cancel_file,wire)
  from shutdown_report import consume
  return run_connected(sockets,wire,TokenClient,plan,epoch,cancel_file,save,stopped_report_loader=lambda role,frame:consume(root,run,role,frame,time.monotonic()+max(.001,epoch+3600-time.time())))
 finally:
  for s in sockets.values():
   try:s.close()
   except OSError:pass
