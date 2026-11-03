"""Six fixed component queries, shared by actual owners and CPU recording."""
def plan():
 return [dict(index=i,phase='warm'if i<2 else'formal',mode=m,past=8496,T=1,candidates=2124,topk=512)for i,m in enumerate([0,1,0,1,1,0])]
def run_owner(rank,create,trial,destroy,shutdown,abort):
 ctx=None;primary=None;cleanup=[];rows=[]
 try:
  ctx=create(rank)
  for item in plan():rows.append(trial(ctx,item))
 except BaseException as exc:
  primary=exc;abort()
 finally:
  if ctx is not None:
   try:destroy(ctx)
   except BaseException as exc:cleanup.append(repr(exc));abort()
  try:shutdown()
  except BaseException as exc:cleanup.append(repr(exc));abort()
 if primary is not None:raise primary
 if cleanup:raise RuntimeError('Owner cleanup failed: '+repr(cleanup))
 return rows
