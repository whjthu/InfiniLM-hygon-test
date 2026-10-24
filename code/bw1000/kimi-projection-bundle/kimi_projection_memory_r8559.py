"""Small read-only memory evidence. Never reclaims cache or changes cgroups."""
import pathlib,os

def read_memory(label):
 import resource
 rows=pathlib.Path('/proc/self/cgroup').read_text().splitlines();records=[]
 for line in rows:
  _,controllers,relative=line.split(':',2)
  if controllers and 'memory'not in controllers.split(','):continue
  p=(pathlib.Path('/sys/fs/cgroup/memory')if controllers else pathlib.Path('/sys/fs/cgroup'))/relative.lstrip('/')
  for q in [p,*list(p.parents)[:4]]:
   if 'job_910375'not in str(q)or not q.is_dir():continue
   values={}
   for name in ['memory.limit_in_bytes','memory.usage_in_bytes','memory.max_usage_in_bytes','memory.failcnt','memory.oom_control','memory.stat','memory.max','memory.current','memory.peak','memory.events']:
    f=q/name
    if f.is_file():values[name]=f.read_text()[:8000]
   record=dict(path=str(q),scope='JOB_ANCESTOR'if q.name=='job_910375'else'CURRENT_WORKER_STEP',files=values);records.append(record)
 info={}
 for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
  name,value=line.split(':',1)
  if name in('MemTotal','MemAvailable'):info[name+'_bytes']=int(value.split()[0])*1024
 job=next((v for v in records if v['scope']=='JOB_ANCESTOR'),None);headroom=None
 if job:
  v=job['files'];limit=v.get('memory.limit_in_bytes',v.get('memory.max'));usage=v.get('memory.usage_in_bytes',v.get('memory.current'))
  if limit and usage and limit.strip().isdigit():headroom=max(0,int(limit)-int(usage))
 return dict(label=label,pid=os.getpid(),step=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),records=records,host_meminfo=info,job_ancestor_current_headroom_bytes=headroom,self_MaxRSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,headroom_scope='Current allocatedjobancestor limit-usage includes chargedcache; not promised96GiB available, cumulativepeak not611specific')
