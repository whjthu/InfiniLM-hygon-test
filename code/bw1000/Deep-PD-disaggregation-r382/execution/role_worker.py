"""One bootstrap-owned main thread, C1 row0, no D prompt replay or automatic retry."""
import time,threading,socket,hashlib,json,traceback
from types import SimpleNamespace
from contract import ROLE_IPS,STATE_PORT,public_request

class RoleWorker:
 def __init__(self,role,leaf,channel,wire,stages,expired,save):
  self.role,self.leaf,self.channel,self.wire,self.stages,self.expired,self.save=role,leaf,channel,wire,stages,expired,save
  self.active=None;self.reports=[];self.parent_thread=threading.get_ident()
 def owner(self):assert threading.get_ident()==self.parent_thread;self.expired()
 def event(self,kind,request,**values):self.channel.send(dict(kind=kind,request_nonce=request['request_nonce'],epoch=request['epoch'],**values))
 def request(self,meta,*,imported_first=None):
  ids=[]if imported_first is None else[imported_first]
  q=SimpleNamespace(max_new_tokens=meta['output_budget'],cancel_event=threading.Event(),abort_reason=None,generated_token_ids=ids)
  def emit(token):
   self.owner();seq=len(ids);ids.append(int(token));self.event('token',meta,seq=seq,token_id=int(token))
  q.emit_token=emit
  if imported_first is None:q.prompt_token_ids=self.leaf.prompt_ids
  return q
 def reason(self,reason,req):
  if reason==self.leaf.reasons.EOS_TOKEN:result='eos'
  elif reason==self.leaf.reasons.LENGTH:result='length'
  else:raise RuntimeError('Canceled/partial stage cannot be published as a completed request')
  assert result=='eos'and req.generated_token_ids[-1]==1 or result=='length'and len(req.generated_token_ids)==req.max_new_tokens
  return result
 def finish(self,meta,req,reason,report):
  result=self.reason(reason,req);report.update(generated_token_ids=list(req.generated_token_ids),finish_reason=result,natural_EOS=result=='eos',quality=self.leaf.quality(req.generated_token_ids))
  self.event('finish',meta,generated_tokens=len(req.generated_token_ids),reason=result)
 def new_report(self,operation,meta):
  self.owner();assert self.active is None
  report=dict(operation=operation,role=self.role,request=public_request(meta),started_local_monotonic=time.monotonic(),normal_reset_before=False,normal_reset_after=False,Native_forward_summary=dict(prefill_calls=0,prefill_input_tokens=0,decode_calls=0),actual_Native=self.leaf.actual_Native)
  self.leaf.reset();self.leaf.stage_started(operation,meta);report['normal_reset_before']=True;self.reports.append(report);self.save(self.reports);return report
 def forward(self,report):
  def call(*args):
   self.owner();kind='decode'if len(args[0])==1 and args[3][0]>0 else'prefill'
   report['Native_forward_summary'][kind+'_calls']+=1
   if kind=='prefill':report['Native_forward_summary']['prefill_input_tokens']+=len(args[0])
   tick=time.monotonic()
   try:return self.leaf.forward(*args)
   finally:report['Native_forward_summary'][kind+'_wall_s']=report['Native_forward_summary'].get(kind+'_wall_s',0.)+time.monotonic()-tick
  return call
 def done(self,operation,meta,report,**extra):
  self.leaf.reset();report.update(normal_reset_after=True,ended_local_monotonic=time.monotonic(),completed=True);self.save(self.reports)
  self.event('stage_done',meta,report=report,**extra)
 def prefill(self,meta,co):
  self.owner();report=self.new_report('co'if co else'prefill',meta);req=self.request(meta);fw=self.forward(report)
  carry=self.stages.prefill_until_first(req,fw,self.leaf.check,self.leaf.finite,self.leaf.reasons,self.expired)
  report.update(prefill_trace=carry['trace'],past_after_prefill=carry['past'],first_token=carry['first_token'],generated_token_ids_before_handoff=list(req.generated_token_ids));self.save(self.reports)
  if not carry['handoff_required']:
   self.finish(meta,req,carry['finish_reason'],report);self.done('co'if co else'prefill',meta,report,early_P_EOS=(not co and carry['first_token']==1));return
  assert carry['past']==8489 and len(req.generated_token_ids)==1
  if co:
   reason,trace=self.stages.decode_from_import(req,carry['past'],carry['first_token'],fw,self.leaf.check,self.leaf.finite,self.leaf.reasons,self.expired)
   report['decode_trace']=trace;self.finish(meta,req,reason,report);self.done('co',meta,report);return
  assert self.role=='P'
  tick=time.monotonic();snapshot=self.leaf.begin_export(meta,carry['past'],carry['first_token']);report['export_snapshot_local_wall_s']=time.monotonic()-tick
  self.active=dict(request=meta,report=report,snapshot=snapshot,carry=carry)
  self.event('prefill_ready',meta,handoff=snapshot.handoff,report=report)
 def transfer(self,meta):
  self.owner();assert self.role=='P'and self.active and self.active['request']==meta
  active=self.active;snapshot=active['snapshot'];sock=socket.socket();tick=time.monotonic()
  try:
   sock.bind((ROLE_IPS['P'],0));self.wire.remaining(sock,self.channel.deadline());sock.connect((ROLE_IPS['D'],STATE_PORT))
   active['report']['transfer']=self.wire.export_state(sock,snapshot.identity,snapshot.blocks,snapshot.read_block,self.channel.deadline())
   snapshot.close();active['report']['source_snapshot_released_after_whole_ticket_ACK']=True
  except BaseException:
   snapshot.abort();raise
  finally:sock.close();active['report']['source_connect_export_ACK_local_wall_s']=time.monotonic()-tick
  self.active=None;self.done('prefill',meta,active['report'])
 def import_decode(self,meta,handoff):
  self.owner();assert self.role=='D';report=self.new_report('import_decode',meta)
  assert handoff['past']==8489 and handoff['first_token']!=1
  tick=time.monotonic();sink=self.leaf.prepare_import(meta,handoff);report['prepare_unpublished_import_local_wall_s']=time.monotonic()-tick
  ticket=self.wire.ImportTicket(sink.identity,sink.blocks,sink);self.event('import_ready',meta)
  sock=None
  try:
   self.wire.remaining(self.channel.state_listener,self.channel.deadline());sock,peer=self.channel.state_listener.accept();assert peer[0]==ROLE_IPS['P']
   tick=time.monotonic();report['transfer']=ticket.receive(sock,self.channel.deadline());report['receive_validate_commit_ACK_local_wall_s']=time.monotonic()-tick
  except BaseException:
   sink.abort();report['Native_import_failure_evidence']=sink.evidence();raise
  finally:
   if sock is not None:sock.close()
  assert sink.group_published,'All8 Native owners must be validated/committed before decode'
  report['all8_import_publish_contract_completed']=True;report['actual_Native_import_published']=bool(self.leaf.actual_Native);report['Native_import_evidence']=sink.evidence()
  req=self.request(meta,imported_first=handoff['first_token']);assert not hasattr(req,'prompt_token_ids')
  reason,trace=self.stages.decode_from_import(req,8489,handoff['first_token'],self.forward(report),self.leaf.check,self.leaf.finite,self.leaf.reasons,self.expired)
  report['decode_trace']=trace;assert report['Native_forward_summary']['prefill_input_tokens']==0
  self.finish(meta,req,reason,report);self.done('import_decode',meta,report)
 def run(self):
  self.channel.send(dict(kind='weights_ready',role=self.role,receipt=self.leaf.receipt))
  try:
   while True:
    self.owner();command=self.channel.receive()
    if command['kind']=='stop':assert self.active is None;break
    assert command['kind']=='stage';meta=command['request'];op=command['operation']
    if op=='co':assert meta['mode']=='co'and meta['co_role']==self.role;self.prefill(meta,True)
    elif op=='prefill':assert self.role=='P'and meta['mode']=='PD';self.prefill(meta,False)
    elif op=='transfer':self.transfer(meta)
    elif op=='import':self.import_decode(meta,command['handoff'])
    else:raise ValueError('Unknown owned-stage command')
   return dict(stage_reports=self.reports,normal_stage_loop_close=True)
  except BaseException as e:
   if self.reports:self.reports[-1].update(error=repr(e),traceback=traceback.format_exc())
   self.save(self.reports);raise
  finally:
   if self.active is not None:self.active['snapshot'].abort();self.active=None
