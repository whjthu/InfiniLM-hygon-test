"""Read actual cgroup memory ancestors only; no weights, CUDA or cache mutation."""
import pathlib,json,os
def memory_guard():
 result={'required_weight_bytes':16102205952,'reserve_bytes':8*1024**3,'ancestors':[],'passed':False,'pid':os.getpid()}
 try:
  cg=pathlib.Path('/proc/self/cgroup').read_text();mountinfo=pathlib.Path('/proc/self/mountinfo').read_text();result['proc_self_cgroup']=cg
  entries=[line.split(':',2)for line in cg.splitlines()]
  v2=[row[2]for row in entries if row[:2]==['0','']]
  v1=[row[2]for row in entries if 'memory'in row[1].split(',')]
  version=2 if v2 else 1;paths=v2 if v2 else v1;assert len(paths)==1,'memory cgroup controller unresolved'
  mounts=[]
  for line in mountinfo.splitlines():
   left,right=line.split(' - ',1);a=left.split();z=right.split()
   if (version==2 and z[0]=='cgroup2')or(version==1 and z[0]=='cgroup'and 'memory'in (z[2].split(',')+a[5].split(',')+[z[1]])):
    def unescape(s):return s.replace('\\040',' ').replace('\\011','\t').replace('\\134','\\')
    mountroot=pathlib.PurePosixPath(unescape(a[3]));cgpath=pathlib.PurePosixPath(paths[0])
    if cgpath.is_relative_to(mountroot):mounts.append((len(mountroot.parts),unescape(a[4]),cgpath.relative_to(mountroot).as_posix(),line))
  assert mounts,'actual memory cgroup mount unresolved';_,point,relative,line=max(mounts)
  result.update(cgroup_version=version,memory_mountinfo=line,memory_controller_path=paths[0])
  root=pathlib.Path(point).resolve();current=(root/relative).resolve();assert current.is_relative_to(root)
  usage='memory.current'if version==2 else'memory.usage_in_bytes';limit='memory.max'if version==2 else'memory.limit_in_bytes';headrooms=[]
  while True:
   row={'path':str(current)}
   for name in [usage,limit]:
    f=current/name;row[name]=f.read_text().strip()if f.is_file()else'NOT_AVAILABLE'
   result['ancestors'].append(row);cap=row[limit];used=row[usage]
   if cap.isdigit():
    assert used.isdigit(),'ancestor memory usage unresolved'
    row['unlimited']=version==1 and int(cap)>=1<<60
    if not row['unlimited']:row['headroom_bytes']=int(cap)-int(used);headrooms.append(row['headroom_bytes'])
   else:assert version==2 and(cap=='max'or current==root),'ancestor memory limit unresolved';row['unlimited']=cap=='max'
   if current==root:break
   current=current.parent
  mem=dict((line.split(':',1)[0],line.split(':',1)[1].strip())for line in pathlib.Path('/proc/meminfo').read_text().splitlines());available=int(mem['MemAvailable'].split()[0])*1024
  result['MemAvailable_bytes']=available;result['minimum_ancestor_headroom_bytes']=min(headrooms)if headrooms else None
  result['admitted_headroom_bytes']=min([available]+headrooms);result['required_total_bytes']=result['required_weight_bytes']+result['reserve_bytes'];result['passed']=result['admitted_headroom_bytes']>=result['required_total_bytes']
 except Exception as exc:result['error']=repr(exc)
 return result
if __name__=='__main__':print(json.dumps(memory_guard()))
