"""Measured scalar progress by default; full raw history only at explicit full-save boundaries."""
import json,time,pathlib,collections
class HostObserver:
 def __init__(self):self.values=collections.defaultdict(float)
 def add(self,name,seconds=0.,size=0):self.values[name+'_s']+=seconds;self.values[name+'_calls']+=1;self.values[name+'_bytes']+=size
 def snapshot(self):return {'at':time.perf_counter(),'values':dict(self.values)}
 def delta(self,base,end=None):
  after=self.snapshot();keys=set(base['values'])|set(after['values']);return {'start_snapshot_monotonic':base['at'],'end_clock_monotonic':after['at']if end is None else end,'counters':{k:after['values'].get(k,0)-base['values'].get(k,0)for k in sorted(keys)},'scope':'local controller perf_counter; StageControl includes barrier/dependencywait, NOT pure networktime; memoryguard includes its StageControl subset, categories overlap and MUST NOT be blindly added; progress IO not GPUkernelwork'}
class TimedControl:
 def __init__(self,control,observer):self.control=control;self.observer=observer
 def __getattr__(self,key):return getattr(self.control,key)
 def collect(self,label,value):
  t=time.perf_counter()
  try:return self.control.collect(label,value)
  finally:
   elapsed=time.perf_counter()-t;self.observer.add('StageControl_collect',elapsed);self.observer.add('StageControl_'+label.split(':',1)[0],elapsed)
class Reporter:
 def __init__(self,path,report,observer):self.path=pathlib.Path(path);self.report=report;self.observer=observer;self.legacy_progress=False;self.progress_path=self.path.with_suffix('.progress.json');self.metrics_path=self.path.with_suffix('.observer.json')
 def _write(self,path,payload,kind,pretty=False):
  tick=time.perf_counter();raw=json.dumps(payload,ensure_ascii=True,indent=2 if pretty else None,separators=None if pretty else(',',':')).encode('utf-8');encoded=time.perf_counter();tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(raw);tmp.replace(path);ended=time.perf_counter();self.observer.add(kind+'_serialize',encoded-tick,len(raw));self.observer.add(kind+'_write',ended-encoded,len(raw))
 def full(self,sidecar=True):
  self.report['host_observer_before_this_fullsave']=self.observer.snapshot();self._write(self.path,self.report,'full_report',True)
  if sidecar:self._write(self.metrics_path,{'stats_after_full_report':self.observer.snapshot(),'scope':'includes lastfullsave; finalsidecar cannot recursivelyincludeitsownwrite, casecriticalpath deltas doinclude all completedsidecarcalls'},'observer_sidecar')
 def progress(self,value):
  self.report['active_batch_progress']=value
  if self.legacy_progress:self.full(False);return
  payload={'schema':1,'node':self.report.get('node'),'run':self.report.get('run'),'phase':self.report.get('phase'),'label':value.get('label'),'steps':value.get('steps'),'active_C':value.get('active_C'),'monotonic':time.perf_counter(),'mode':'tiny_progress'};self._write(self.progress_path,payload,'scalar_progress')
 def set_logging_diag(self,enabled):assert type(enabled)is bool;self.legacy_progress=enabled
