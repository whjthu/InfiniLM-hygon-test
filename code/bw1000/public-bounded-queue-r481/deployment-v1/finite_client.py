"""Finite actual open-arrival client. No retries, gold gates or hidden rows."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,http.client,json,socket,threading,time,urllib.parse
# Passed frozen r480 source directory must already be on PYTHONPATH.
from client_clock import ClientClock
from scoring import grade
H=lambda b:hashlib.sha256(b).hexdigest()
def main(config):
 cfg=json.loads(Path(config).read_bytes());raw=Path(cfg['private_plan']).read_bytes();assert H(raw)==cfg['private_plan_SHA256'];plan=json.loads(raw);assert len(plan['requests'])==20 and plan['ordinals']==[0,1,128,129]
 server=json.loads(Path(cfg['server_config']).read_bytes());assert H(Path(cfg['server_config']).read_bytes())==cfg['server_config_SHA256']
 from public_deep_model.protocol import Run
 ready=Run(server).read('MODEL_READY.json');assert ready['Model_READY']and ready['Native_internal_owner_count']==8 and ready['HC_mode']==0 and ready['Native_SHA256']=='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'
 assert ready['fixed_profile']['fixed_mode']==ready['fixed_profile']['r366_fixed']==0 and ready['fixed_profile']['HC_groups']==86 and ready['queue_policy']['enabled']==cfg['queue_policy']
 out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);base=urllib.parse.urlsplit(cfg['base_url']);assert base.scheme=='http'and base.hostname
 result=dict(completed=False,planned=20,warm=0,rows=[dict(index=i,sent=False,completed=False,missing=True)for i in range(20)],retries=0,pair=cfg['pair'],queue_policy=cfg['queue_policy']);lock=threading.Lock();acks=[threading.Event()for _ in range(20)];admission_observed=[False]*20;ack_reasons=[[]for _ in range(20)];all_last=threading.Event()
 def save():
  with lock:
   tmp=out/'result.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(out/'result.json')
 def ack(i,observed=True,reason='role_or_error'):
  if reason not in ack_reasons[i]:ack_reasons[i].append(reason)
  admission_observed[i]=admission_observed[i]or observed;acks[i].set()
  if all(acks[j].is_set()for j in range(12,20)):all_last.set()
 def stop():
  # The original typed Run validates exactrun/step/source/config identity.
  from public_deep_model.protocol import Run
  server=json.loads(Path(cfg['server_config']).read_bytes());Run(server).stop();result['STOP_ack_snapshot']=[dict(index=i,admission_observed=admission_observed[i],reasons=list(ack_reasons[i]))for i in range(20)];result['STOP_trigger_scope']='all_last_actual_role_or_rejection'if all(admission_observed[j]for j in range(12,20))else'all_transports_finished_or_original_deadline; unknown admission retained';result['STOP_requested_epoch']=time.time();save()
 def request(row,phase_start=None):
  i=row['index'];r=dict(index=i,case_id=row['case_id'],ordinal=row['ordinal'],sent=False,completed=False,missing=True,planned_cancel=None);connection=None
  if phase_start is not None:time.sleep(max(0,phase_start+row['arrival_offset_s']-time.monotonic()))
  if time.time()>=cfg['deadline_unix']:return dict(r,error='Original residency deadline before send')
  body=json.dumps(row['body'],ensure_ascii=False,separators=(',',':')).encode();assert H(body)==row['actual_body_SHA256'];(out/('%02d.request.json'%i)).write_bytes(body);start=time.monotonic_ns();clock=ClientClock(start);cancel_at=None;reads=[];seen_frames=0;saw_role=False;saw_error=False
  try:
   r.update(sent=True,missing=False,request_start_ns=start,actual_body_SHA256=H(body),prompt_IDs_SHA256=row['prompt_IDs_SHA256']);connection=http.client.HTTPConnection(base.hostname,base.port or 80,timeout=max(.01,cfg['deadline_unix']-time.time()));connection.request('POST',(base.path.rstrip('/')or'')+'/chat/completions',body,{'Content-Type':'application/json','x-infinilm-token-clock':'1'});sock=connection.sock;response=connection.getresponse();clock.begin_response(response.status);r.update(HTTP_status=response.status,headers_ns=time.monotonic_ns())
   with(out/('%02d.wire.bin'%i)).open('xb')as wire:
    if response.status!=200:
     b=response.read();wire.write(b);r.update(rejected=True,fully_received_EOF=True,error_body_SHA256=H(b));ack(i,True,'HTTP_reject')
    else:
     while True:
      remaining=cfg['deadline_unix']-time.time()
      if cancel_at is not None:remaining=min(remaining,cancel_at-time.monotonic())
      if remaining<=0:
       r['planned_cancel']='queued_after_role'if cfg['pair']=='B'and i in[5,7,9,11]else'active_after_token'if cfg['pair']=='B'and i==4 else'original_deadline';break
      sock.settimeout(max(.001,remaining));b=response.read1(65536);tick=time.monotonic_ns();wire.write(b);reads.append(dict(offset=wire.tell()-len(b),bytes=len(b),received_ns=tick));clock.feed(b,tick,final=not b)
      for f in clock.frames[seen_frames:]:
       if f['data']=='[DONE]':continue
       value=json.loads(f['data'])
       if value.get('id')is not None:
        rid=value['id'];assert isinstance(rid,str)and rid and r.get('server_request_id',rid)==rid,'Inconsistent actual SSE requestID';r['server_request_id']=rid
       saw_role=saw_role or any(c.get('delta',{}).get('role')for c in value.get('choices',[]));saw_error=saw_error or bool(value.get('error'))
      seen_frames=len(clock.frames);role=saw_role;errors=saw_error
      if role or errors:ack(i,True,'role'if role else'error_frame')
      if cfg['pair']=='B'and i in[5,7,9,11]and role and cancel_at is None:cancel_at=time.monotonic()+.15
      if cfg['pair']=='B'and i==4 and clock.tokens:r['planned_cancel']='active_after_token';break
      if not b:r['fully_received_EOF']=True;break
   r.update(completed=True,clock=clock.summary(),frames=clock.frames,read_clock=reads)
  except BaseException as error:
   r.update(error=repr(error),clock=clock.summary(),frames=clock.frames,read_clock=reads)
   if cancel_at is not None and time.monotonic()>=cancel_at:r['planned_cancel']='queued_after_role'
   if time.time()>=cfg['deadline_unix']:r['original_deadline_hit']=True
  finally:
   if connection is not None:connection.close()
   r['ended_ns']=time.monotonic_ns();r['transport_terminated']=True;r.setdefault('server_request_id',None);r['server_request_id_scope']='actual SSE data.id only; missing UNKNOWN, never inferred from arrival/marker';ack(i,False,'transport_terminal')
   wire=out/('%02d.wire.bin'%i)
   if wire.is_file():r['wire_SHA256']=H(wire.read_bytes())
  return r
 def guarded(row,phase_start=None):
  try:return request(row,phase_start)
  except BaseException as error:
   ack(row['index'],False,'transport_terminal');return dict(index=row['index'],case_id=row['case_id'],ordinal=row['ordinal'],sent=False,completed=False,missing=True,transport_terminated=True,error=repr(error),outcome='Client infrastructure failed; admission UNKNOWN')
 save()
 try:
  for row in plan['requests'][:4]:result['rows'][row['index']]=guarded(row);save()
  waves=[plan['requests'][4:]]if cfg['pair']=='A'else[plan['requests'][4:12],plan['requests'][12:20]]
  result['waves']=[]
  for wave in waves:
   start=time.monotonic()-wave[0]['arrival_offset_s'];result['waves'].append(dict(indices=[r['index']for r in wave],start_ns=time.monotonic_ns()))
   with concurrent.futures.ThreadPoolExecutor(max_workers=17)as pool:
    futures={pool.submit(guarded,row,start):row['index']for row in wave}
    def stop_backlog():
     if all_last.wait(max(0,cfg['deadline_unix']-time.time())):stop()
    final=pool.submit(stop_backlog)if cfg['pair']=='B'and wave[0]['index']==12 else None
    for future in concurrent.futures.as_completed(futures):result['rows'][futures[future]]=future.result();save()
    if final is not None:final.result()
 finally:
  # Quality failures are collected, never a controller TERM/KILL gate.
  if'STOP_requested_epoch'not in result:stop()
  lookup={r['case_id']:r['gold']for r in plan['records']}
  for row in result['rows']:
   content='';tool=False;fin=None
   for frame in row.get('frames',[]):
    if frame['data']=='[DONE]':continue
    event=json.loads(frame['data'])
    for choice in event.get('choices',[]):
     delta=choice.get('delta',{});content+=delta.get('content')or'';tool=tool or bool(delta.get('tool_calls'));fin=choice.get('finish_reason')or fin
   planned=plan['requests'][row['index']];row.setdefault('case_id',planned['case_id']);clock=row.get('clock',{});natural=bool(clock.get('EOS_tokens')==1 and clock.get('owned_finished')and row.get('fully_received_EOF')and not row.get('planned_cancel'))
   row['score']=grade(lookup[row['case_id']],content,natural_EOS=natural,cap_hit=bool(clock.get('capped')or fin=='length'),unexpected_tool=tool);row['content_SHA256']=H(content.encode())
  result['ack_reasons']=[dict(index=i,admission_observed=admission_observed[i],reasons=ack_reasons[i])for i in range(20)]
  for row in result['rows']:row['unsent']=not row['sent'];row['incomplete']=not row.get('clock',{}).get('complete',False)
  result['missing_scope']='legacy missing means unsent only; incomplete is separately preserved'
  result['incomplete']=sum(r['incomplete']for r in result['rows'])
  result['quality_summary']=dict(planned=20,correct=sum(r['score']['correct']for r in result['rows']),natural_EOS=sum(r['score']['natural_EOS']for r in result['rows']),cap=sum(r['score']['cap_hit']for r in result['rows']),parse_failure=sum(r['score']['parse_failure']for r in result['rows']),no_filter=True)
  result.update(completed=True,sent=sum(r['sent']for r in result['rows']),terminal=sum(r.get('transport_terminated',False)for r in result['rows']),missing=sum(r['missing']for r in result['rows']),quality='Original independent fixedgold scores with all20 planned rows; no correctness lifecycle gate');save()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();main(a.config)
