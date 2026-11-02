"""Fixed empty-context fullphase comparison, not a cache or Model benchmark."""
import time,traceback

def plan():
 return [dict(ordinal=i*10+j,L=L,phase='warm'if j<2 else'formal',discard_raw=flag,arm='ON'if flag else'OFF',position=j)
  for i,L in enumerate([8635,32638])for j,flag in enumerate([False,True,False,True,True,False,False,True,True,False])]

def expected(spec):
 n=(spec['L']+511)//512
 return dict(queries=8,committed_appends=7,prompt_appends=n,commands=n+19,fences=n+23,feature_bytes=(spec['L']+56)*35840*2)

def run_phase(session,body,spec,*,remaining=lambda:None,clock=time.perf_counter,save=lambda q:None):
 c=session.context;q=dict(**spec,completed=False,passed=False,commands=[],queries=[],profile=False,
  scope='Complete begin+512prefix+seal+8query/7syntheticcommitted8+retire+close. Same-input geometry, no target acceptance or Model throughput.')
 capture=0;start=clock();save(q)
 def command(cmd,data=b'',**fields):
  remaining();meta=dict(cmd=cmd,seq=session.sequence)
  if cmd!='close':meta.update(epoch=1,request=1)
  meta.update(fields);tick=clock();answer=session.dispatch(meta,data)
  q['commands'].append(dict(cmd=cmd,past=fields.get('past'),rows=fields.get('rows'),body_bytes=len(data),wall_s=clock()-tick))
  return answer
 def append(cmd,past,n):
  nonlocal capture
  capture+=1;fields=dict(past=past,rows=n,capture_epoch=capture)
  if cmd=='append_committed':fields['target_commit']=dict(capture_epoch=capture,past_before=past,past_after=past+n,accepted_inputs=n,complete=True)
  return command(cmd,body(past,n),**fields)
 try:
  tick=clock();command('begin',prompt_tokens=spec['L'])
  for past in range(0,spec['L'],512):append('append_prompt',past,min(512,spec['L']-past))
  command('seal',past=c.length);q['prefix_begin_append_seal_wall_s']=clock()-tick
  tick=clock()
  for cycle in range(8):
   if cycle:append('append_committed',c.length,8)
   assert c.length==spec['L']+8*cycle and c.phase=='decode'
   answer=command('query',past=c.length,anchor=163584,K=7)
   assert answer['mode']=='draft'and len(answer['proposals'])==7
   assert all(type(v)is int and 0<=v<163840 for v in answer['proposals'])
   q['queries'].append(dict(cycle=cycle,past=c.length,proposals=answer['proposals'],handler_wall_s=answer['handler_wall_s']))
  q['eight_query_and_seven_append_wall_s']=clock()-tick
  q['payload_before_retire']=c.payload_bytes();assert (c.raw is None)==spec['discard_raw']
  tick=clock();assert command('retire')['removed']is True;assert command('close')['closed']is True
  assert c.raw is None and c.projected is None and session.closed
  q['retire_close_wall_s']=clock()-tick
  q['complete_phase_wall_s']=clock()-start
  q.update(normal_session_close=True,passed=True,query_dispatch_sum_s=sum(v['wall_s']for v in q['commands']if v['cmd']=='query'),committed_append_dispatch_sum_s=sum(v['wall_s']for v in q['commands']if v['cmd']=='append_committed'))
  assert len(q['commands'])==expected(spec)['commands']and len(q['queries'])==8
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc());save(q);raise
 finally:
  try:
   if not session.closed:session.close()
  except BaseException as e:q.update(passed=False,cleanup_error=repr(e));raise
  finally:q['completed']=True;save(q)
 return q
