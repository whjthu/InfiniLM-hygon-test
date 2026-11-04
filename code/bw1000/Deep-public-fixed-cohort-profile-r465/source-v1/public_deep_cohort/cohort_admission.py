"""Same-process NativeRequests; no IPC or continuous admission/refill."""
from .profile import ticket
def add(engine,original,kwargs):
 t=ticket(kwargs.get('request_data')or{});c=engine._fixed_ticket
 if c is not None and(c['id']!=t['id']or c['mode']!=t['mode']or t['row']in c['rows']):raise ValueError('Foreign/duplicate row while closed cohort active')
 req=original(**kwargs)
 if c is None:c=engine._fixed_ticket=dict(id=t['id'],mode=t['mode'],rows=set())
 c['rows'].add(t['row']);req.metadata['native_fixed_cohort']=t
 assert req.max_new_tokens==4096 and len(req.prompt_token_ids)+4096<=32768
 return req
def release(engine,identifier):
 with engine._admission_lock:
  assert engine._fixed_ticket is not None and engine._fixed_ticket['id']==identifier
  engine._fixed_ticket=None
