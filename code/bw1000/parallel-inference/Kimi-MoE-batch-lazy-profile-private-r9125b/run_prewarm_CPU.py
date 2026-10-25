"""Bounded CPU-only selected-node page prewarm; no clones/SHA/GPU/pinning claim."""
import pathlib,json,os,time,hashlib,signal,resource
signal.alarm(140)
r=pathlib.Path('/refactor');node=int(os.environ['SLURM_PROCID']);tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];plan=json.loads(pathlib.Path(__file__).with_name('prewarm-ranges-input-r9125.json').read_text());source=pathlib.Path('/shared/Kimi-K3')/plan['file_name'];out=r/'results'/('Kimi-MoE-lazy-profile-r9125b-prewarm-'+tag+'-node'+str(node)+'.json');assert not out.exists();page=os.sysconf('SC_PAGE_SIZE');ranges=[];payload=plan['payload_offset'];filebytes=plan['source_file_bytes']
with source.open('rb')as f:n=int.from_bytes(f.read(8),'little');hdr=f.read(n)
assert hashlib.sha256(hdr).hexdigest()==plan['source_header_SHA256']and source.stat().st_size==filebytes
def add(a,b):ranges.append((a//page*page,min(filebytes,(b+page-1)//page*page)))
for item in plan['per_tensor']:
 shape=item['full_shape'];lo,hi=item['file_data_offsets'];dim=item['tp_dim'];base=payload+lo
 if dim is None:add(base,payload+hi);continue
 a,b=item['node'+str(node)+'_axis_range']
 if dim==0:
  stride=(hi-lo)//shape[0];add(base+a*stride,base+b*stride)
 else:
  assert len(shape)==2;stride=(hi-lo)//shape[0];bpe=stride//shape[1];gap=stride-(b-a)*bpe
  if gap<page:add(base,payload+hi)
  else:
   for row in range(shape[0]):add(base+row*stride+a*bpe,base+row*stride+b*bpe)
ranges.sort();merged=[]
for a,b in ranges:
 if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
 else:merged.append([a,b])
def snap():
 p=pathlib.Path('/proc/self');u=resource.getrusage(resource.RUSAGE_SELF);return {'io':{a:int(b)for a,b in(v.split(':')for v in(p/'io').read_text().splitlines())},'maxRSS_kB':u.ru_maxrss,'CPU_seconds':u.ru_utime+u.ru_stime}
report={'producer':tag,'node':os.uname().nodename,'CPU_only':True,'GPU_called':False,'completed':False,'planned_logical_union_bytes':plan['actual_needed_logical_union_bytes_per_node'],'planned_page_read_bytes':sum(b-a for a,b in merged),'merged_intervals':len(merged),'node_rank':node,'checkpoint_header_SHA256':plan['source_header_SHA256'],'physical_page_overfetch_explanation':'W2selectedhalfcolumns touchevery4Kpage whenremainingrowgap<pagesize; page readbytes isnotlogical halfbytes','pinned_or_full_CPU_residency_proven':False,'CPU_clones':0,'new_full_tensor_SHA':False,'before':snap(),'read_completed_bytes':0};start=time.perf_counter();fd=os.open(source,os.O_RDONLY);budget=120;checksum=0
try:
 for begin,end in merged:
  offset=begin
  while offset<end:
   if time.perf_counter()-start>budget:raise TimeoutError('CPU prewarm120s budget; no GPU allocated')
   chunk=os.pread(fd,min(8<<20,end-offset),offset);assert chunk;offset+=len(chunk);report['read_completed_bytes']+=len(chunk);checksum^=chunk[0];del chunk
 report['completed']=True
except BaseException as e:report['error']=repr(e)
finally:os.close(fd)
report.update(after=snap(),wall_seconds=time.perf_counter()-start,bounded_firstbyte_XOR_not_integrity_hash=checksum,interpretation='Currentcache prewarm cost only; sourcefile readonly. OS pagecache mayevict afterCPUstep, notpinnedmemory and notModel/H2D speedup. No coldcache eviction, no GPU re-submit.');out.write_text(json.dumps(report,indent=2));print(json.dumps({'producer':tag,'node':report['node'],'completed':report['completed'],'wall_seconds':report['wall_seconds'],'path':str(out),'SHA256':hashlib.sha256(out.read_bytes()).hexdigest()}))
