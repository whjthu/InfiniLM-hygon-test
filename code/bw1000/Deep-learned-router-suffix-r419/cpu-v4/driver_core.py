"""Fixed future component schedule; actual same-logits qualification precedes formal work."""
def plan():return [dict(index=i,phase='warm'if i<2 else'formal',mode=m)for i,m in enumerate([0,1,0,1,1,0,1,0,0,1])]
def run_owner(rank,create,initialize,qualify,trial,destroy,shutdown,abort):
 ctx=None;primary=None;cleanup=[];rows=[]
 def stop():
  try:abort()
  except BaseException as exc:cleanup.append("abort:"+repr(exc))
 try:
  ctx=create(rank);initialize(ctx);qualify(ctx)
  for item in plan():rows.append(trial(ctx,item))
 except BaseException as exc:primary=exc;stop()
 finally:
  if ctx is not None:
   try:destroy(ctx)
   except BaseException as exc:cleanup.append(repr(exc));stop()
  try:shutdown()
  except BaseException as exc:cleanup.append(repr(exc));stop()
 if primary is not None:
  if cleanup:setattr(primary,'cleanup_errors',cleanup)
  raise primary
 if cleanup:raise RuntimeError('r419 owner cleanup failed:'+repr(cleanup))
 return rows
