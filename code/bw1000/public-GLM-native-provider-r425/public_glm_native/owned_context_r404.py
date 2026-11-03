"""One already-created private GLM ctx; never creates or retries a Context."""
from pathlib import Path
import json,os,traceback
class OwnedContext:
 def __init__(self,lib,ctx,check,stage,rank,path=None):
  self.lib,self.ctx,self.check=lib,ctx,check
  self.path=Path(path)if path else Path('/refactor/results')/('GLM-r404-lifecycle-'+os.environ['GLM_RUN_R5040']+'-stage%d-rank%d.json'%(stage,rank))
  self.q=dict(completed=False,normal=False,stage=stage,rank=rank,run=os.environ['GLM_RUN_R5040'],actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],source_manifest_SHA256=os.environ['GLM_SOURCE_MANIFEST_SHA'],constructor_calls_expected=1,existing_context_only=True,destroy_attempts=0,shutdown_attempts=0,cleanup_errors=[],primary_error=None)
 def save(self):
  try:
   temp=self.path.with_suffix('.tmp');temp.write_text(json.dumps(self.q,indent=2)+'\n');temp.replace(self.path);return True
  except OSError as e:self.q['record_write_error']=repr(e);return False
 def __enter__(self):
  try:
   if self.path.exists():
    self.path=self.path.with_name(self.path.stem+'-rejected-existing-'+str(os.getpid())+'.json')
    raise RuntimeError('r404 lifecycle record already exists; no replay')
   if not self.save():raise RuntimeError('r404 lifecycle initial record could not be written')
   return self
  except BaseException as e:self.__exit__(type(e),e,e.__traceback__);raise
 def destroy(self,ctx):
  if ctx is not self.ctx and ctx!=self.ctx:raise RuntimeError('r404 foreign ctx destroy')
  if self.q['destroy_attempts']:raise RuntimeError('r404 duplicate destroy refused')
  self.q['destroy_attempts']=1;self.save()
  try:self.q['destroy_RC']=int(self.lib.modern_glm_model_destroy(self.ctx));return self.q['destroy_RC']
  except BaseException as e:self.q['destroy_error']=repr(e);raise
  finally:self.save()
 def shutdown(self):
  if self.q['shutdown_attempts']:raise RuntimeError('r404 duplicate shutdown refused')
  self.q['shutdown_attempts']=1;self.save()
  try:self.q['shutdown_RC']=int(self.lib.modern_glm_model_shutdown());return self.q['shutdown_RC']
  except BaseException as e:self.q['shutdown_error']=repr(e);raise
  finally:self.save()
 def __exit__(self,kind,error,tb):
  self.q['primary_error']=repr(error)if error is not None else None
  if error is not None:self.q['primary_error_frames']=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name)for f in traceback.extract_tb(tb)[-6:]]
  for name,args in [('destroy',(self.ctx,)),('shutdown',())]:
   if self.q[name+'_attempts']:continue
   try:self.check(getattr(self,name)(*args))
   except BaseException as e:self.q['cleanup_errors'].append(dict(operation=name,error=repr(e)))
  for name in ['destroy','shutdown']:
   if self.q.get(name+'_RC')not in(0,None)or name+'_error'in self.q:
    if not any(e['operation']==name for e in self.q['cleanup_errors']):self.q['cleanup_errors'].append(dict(operation=name,rc=self.q.get(name+'_RC'),error=self.q.get(name+'_error')))
  self.q.update(completed=True,normal=error is None and not self.q['cleanup_errors']and not self.q.get('record_write_error')and self.q.get('destroy_RC')==self.q.get('shutdown_RC')==0)
  if not self.save():self.q['normal']=False;print(json.dumps(dict(r404_context_cleanup_record_failed=self.q)),flush=True)
  if error is None and not self.q['normal']:raise RuntimeError('r404 Context cleanup failed: '+repr(self.q['cleanup_errors']))
  return False
