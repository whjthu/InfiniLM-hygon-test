"""C1 two-role coordinator. The two cohort requests overlap only across P and D."""
import time,copy,queue,hashlib
from contract import request_groups,public_request

class Scheduler:
 def __init__(self,plan,send,TokenClient,nonce,clock=time.monotonic,check=lambda:None,save=lambda value:None,client_deadline=None):
  self.plan,self.send,self.TokenClient,self.clock,self.check,self.save=plan,send,TokenClient,clock,check,save
  self.groups=request_groups(plan,nonce);self.clients={};self.rows={};self.busy={'P':None,'D':None};self.events=queue.Queue();self.reports=[]
  self.current=None;self.reader_error=None;self.client_deadline=client_deadline
 def receive(self,role,event):
  # Called immediately on each authenticated socket reader. TokenClient records
  # this one client clock and any true cross-role reordering/visible delay.
  if event.get('kind')in ['token','finish']:
   key=event['request_nonce'];assert key in self.clients,'Unadmitted request emitted a token'
   self.clients[key].frame(role,event)
  else:self.events.put((role,event))
 def failure(self,role,error):self.events.put((role,dict(kind='error',error=repr(error))))
 def _send_stage(self,role,operation,request,**values):
  key=request['request_nonce'];row=self.rows[key]
  if operation!='transfer':assert self.busy[role]is None;self.busy[role]=key
  else:assert self.busy[role]==key
  row.setdefault('commands',[]).append(dict(role=role,operation=operation,client_send_monotonic=self.clock(),other_role_busy_at_dispatch=self.busy['D'if role=='P'else'P']is not None))
  self.send(role,dict(kind='stage',operation=operation,request=request,**values))
 def _drain(self,timeout):
  try:role,event=self.events.get(timeout=timeout)
  except queue.Empty:return
  if event['kind']=='error':raise RuntimeError('Owned '+role+' stage failed: '+event['error'])
  key=event['request_nonce'];assert key in self.rows
  row=self.rows[key];assert event['epoch']==row['request']['epoch']and self.busy[role]==key
  kind=event['kind'];row.setdefault('events',[]).append(dict(role=role,kind=kind,client_received_monotonic=self.clock()))
  if kind=='prefill_ready':
   assert role=='P'and row['request']['mode']=='PD'and not row.get('prefill_ready')
   row['prefill_ready']=True;row['handoff']=event['handoff'];row['P_prefill_report']=event['report']
  elif kind=='import_ready':
   assert role=='D'and row.get('import_started')and not row.get('transfer_started')
   row['transfer_started']=True;self._send_stage('P','transfer',row['request'])
  elif kind=='stage_done':
   self.busy[role]=None;row[role+'_done']=True;row[role+'_report']=event['report']
   if event.get('early_P_EOS'):row['D_skipped_early_P_EOS']=True
  else:raise ValueError('Unexpected stage coordinator event: '+kind)
 def _schedule(self,group,started):
  now=self.clock()
  for req in group['requests']:
   key=req['request_nonce']
   if key not in self.rows and now>=started+req['arrival_offset_s']:
    row=dict(request=req,scheduled_arrival_monotonic=started+req['arrival_offset_s'],actual_admission_monotonic=self.clock(),commands=[],events=[])
    self.rows[key]=row
    self.clients[key]=self.TokenClient(key,req['epoch'],req['mode'],req['co_role'],req['output_budget'],admitted=row['actual_admission_monotonic'],deadline=self.client_deadline)
  for req in group['requests']:
   key=req['request_nonce']
   if key not in self.rows:continue
   row=self.rows[key]
   if not row.get('started'):
    role=req['co_role']if req['mode']=='co'else'P'
    if self.busy[role]is None:
     row['started']=True;self._send_stage(role,'co'if req['mode']=='co'else'prefill',req)
   if req['mode']=='PD'and row.get('prefill_ready')and not row.get('import_started')and self.busy['D']is None:
    row['import_started']=True;self._send_stage('D','import',req,handoff=row['handoff'])
 def _complete(self,req):
  key=req['request_nonce']
  if key not in self.rows or not self.clients[key].summary()['completed']:return False
  r=self.rows[key]
  return bool(r.get(req['co_role']+'_done'))if req['mode']=='co'else bool(r.get('P_done')and(r.get('D_done')or r.get('D_skipped_early_P_EOS')))
 def snapshot(self):
  rows=[]
  for group in self.groups:
   for req in group['requests']:
    key=req['request_nonce'];r=self.rows.get(key)
    row=dict(request=public_request(req),admitted=r is not None)
    if r is not None:
     for name in ['commands','events','scheduled_arrival_monotonic','actual_admission_monotonic','P_report','D_report','P_prefill_report','D_skipped_early_P_EOS']:
      if name in r:row[name]=copy.deepcopy(r[name])
     client=self.clients[key];summary=client.summary();summary.pop('request_nonce');row['client']=summary
     with client.lock:row['client_token_ID_timeline']=copy.deepcopy(client.delivered)
    rows.append(row)
  return dict(planned_external=len(rows),requests=rows,groups=copy.deepcopy(self.reports),client_clock='One controller monotonic clock; token frame processing occurs in authenticated reader callbacks',HTTP_SSE_SLO=False)
 def run(self):
  for group in self.groups:
   self.current=group['id'];started=self.clock()
   while True:
    self.check();self._schedule(group,started)
    if all(self._complete(req)for req in group['requests']):break
    self._drain(.005)
   assert self.busy=={'P':None,'D':None}
   self.reports.append(dict(id=group['id'],phase=group['phase'],mode=group['mode'],requests=len(group['requests']),client_group_start_monotonic=started,client_group_end_monotonic=self.clock(),all_planned_requests_terminal=True))
   if group['phase']=='warm':
    warm=[]
    for req in group['requests']:
     row=self.rows[req['request_nonce']];source=row[req['co_role']+'_report']if req['mode']=='co'else row['P_report']if row.get('D_skipped_early_P_EOS')else row['D_report']
     warm.append(dict(index=req['index'],task_correct=source['quality']['task_correct'],natural_EOS=source['natural_EOS']))
    self.reports[-1]['warm_quality_guard']=warm
    if not all(r['task_correct']and r['natural_EOS']for r in warm):
     result=self.snapshot();result.update(completed=False,warm_gate_failed=True,normal_stop_after_failed_warm=True);self.save(result);return result
   self.save(self.snapshot())
  result=self.snapshot();assert len(self.clients)==self.plan['planned_external_requests']
  result['completed']=True;return result
