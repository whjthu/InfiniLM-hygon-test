"""One authenticated production Session; OFF delegates unchanged362b serving."""
from policy import OFF,ON,checked,ready

def serve_authenticated(old,listener,hello,runtime_factory,report,save,mode=OFF):
 checked(mode)
 try:
  if mode==OFF:return old.serve_authenticated(listener,hello,runtime_factory,report,save)
  from batch_wire_r9128 import recv,send
  conn=runtime=None
  try:
   conn,_=listener.accept();conn.settimeout(120)
   meta,body=recv(conn)
   if meta!=hello or body:raise ValueError('Exact r388 ON policy/nonce/source identity required')
   report['authenticated']=True;save()
   runtime=runtime_factory()
   send(conn,ready(old,mode,hello));report['ready']=True;save()
   while not runtime.session.closed:
    meta,body=recv(conn)
    if report.get('commands',0)>=4096:raise ValueError('Bounded command count exceeded')
    answer=runtime.session.dispatch(meta,body)
    report['commands']=report.get('commands',0)+1;report['feature_bytes']=report.get('feature_bytes',0)+len(body)
    counts=report.setdefault('phase_counts',{});cmd=meta['cmd'];counts[cmd]=counts.get(cmd,0)+1
    report['handler_wall_s']=report.get('handler_wall_s',0.)+answer.get('handler_wall_s',0.)
    send(conn,answer)
    if cmd in ['retire','close']:save()
   report['normal_protocol_close']=True
  finally:
   try:
    if runtime is not None:runtime.close()
   finally:
    try:
     if conn is not None:conn.close()
    finally:listener.close()
 finally:runtime_factory.close_scope()
