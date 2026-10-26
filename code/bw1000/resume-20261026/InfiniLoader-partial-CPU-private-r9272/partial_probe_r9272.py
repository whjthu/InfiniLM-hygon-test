"""Bounded CPU-only original mmap vs one sealed memfd shard; no GPU initialization."""
import os,sys,json,time,mmap,hashlib,struct,subprocess,threading,re,resource,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from infiniloader.memory import memory_info
from infiniloader.runtime import process_token
from infiniloader.ipc import request
SRC=Path('/public/home/aclsylqidf/shared/Kimi-K3')
SHARD='model-00001-of-000096.safetensors'
GIB=1024**3

def header(path):
 with open(path,'rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];assert 0<n<64*1024**2
  b=f.read(n);assert len(b)==n
 return b,json.loads(b),8+n

def mapped_memory(pid,inode):
 rows=[];current=None
 for line in Path('/proc/%d/smaps'%pid).read_text().splitlines():
  if re.match(r'^[0-9a-f]+-[0-9a-f]+ ',line):
   fields=line.split(None,5); current={'mapping':line} if int(fields[4])==inode else None
   if current is not None:rows.append(current)
  elif current is not None and ':' in line:
   k,v=line.split(':',1)
   if v.strip().split() and v.strip().split()[0].isdigit():current[k]=int(v.strip().split()[0])*1024
 return {'mappings':rows,'sum':{k:sum(x.get(k,0) for x in rows) for k in ['Rss','Pss','Shared_Clean','Shared_Dirty','Private_Clean','Private_Dirty','Swap']}}

def reader(path,socket):
 lease=None
 try:
  if socket!='none':lease=request(socket,{'command':'lease-acquire','pid':os.getpid(),'process_token':process_token(os.getpid())})['lease']
  t=time.perf_counter();hb,h,base=header(path)
  with open(path,'rb') as f:
   mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
   before=resource.getrusage(resource.RUSAGE_SELF);t_touch=time.perf_counter();checksum=sum(mm[i] for i in range(0,len(mm),4096));touched=time.perf_counter()-t_touch
   selected=[]
   for key,row in sorted(h.items()):
    if key=='__metadata__':continue
    a,b=row['data_offsets'];assert 0<=a<=b<=len(mm)-base
    if 0<b-a<=4*1024**2 and len(selected)<3:selected.append({'name':key,'shape':row['shape'],'dtype':row['dtype'],'bytes':b-a,'sha256':hashlib.sha256(mm[base+a:base+b]).hexdigest()})
   assert selected,'need bounded real tensors for byte identity'
   after=resource.getrusage(resource.RUSAGE_SELF)
   compat={'available':False}
   try:
    from safetensors import safe_open
   except ImportError:pass
   else:
    with safe_open(str(path),framework='pt',device='cpu') as sf:
     keys=set(sf.keys());assert keys==set(h)-{'__metadata__'}
     for x in selected:assert list(sf.get_slice(x['name']).get_shape())==x['shape']
     z=sf.get_tensor(selected[0]['name']);assert list(z.shape)==selected[0]['shape']
     compat={'available':True,'keys':len(keys),'selected_shapes':True,'small_tensor_materialized_cpu':True};del z
   row={'pid':os.getpid(),'path':str(path),'inode':os.fstat(f.fileno()).st_ino,'header_sha256':hashlib.sha256(hb).hexdigest(),'tensor_count':len(h)-('__metadata__' in h),'selected':selected,'page_touch_seconds':touched,'open_header_touch_identity_seconds':time.perf_counter()-t,'page_checksum':checksum,'minor_fault_delta':after.ru_minflt-before.ru_minflt,'major_fault_delta':after.ru_majflt-before.ru_majflt,'safe_open':compat}
   print(json.dumps(row),flush=True);sys.stdin.readline();mm.close()
 finally:
  if lease:request(socket,{'command':'lease-release','lease':lease})

def daemon(work):
 from infiniloader.daemon import WeightLoaderDaemon
 work=Path(work);run=work/'run';run.mkdir(exist_ok=True)
 d=WeightLoaderDaemon(name='partial-r9272',source=SRC,view=work/'cached-view',socket_path=run/'d.sock',pid_path=run/'d.pid',state_path=run/'d.state.json',upstream_source=None,local_cache_path=None,include_patterns=(SHARD,),exclude_patterns=(),replace=False,force=False,max_memory_fraction=.9,chunk_size=64*1024**2,workers=1,max_memory_bytes=16*GIB,reserve_bytes=8*GIB,max_resident_bytes=16*GIB,idle_ttl=120,quiet=False)
 return d.serve_forever()

def main(work):
 work=Path(work).resolve();assert not work.exists();work.mkdir(mode=0o700)
 out={'schema':1,'scope':'CPU_COMPONENT_ONLY_WARM_PAGECACHE_NO_COLD_OR_MODEL_SPEED_CLAIM','host':os.uname().nodename,'step':os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),'source':str(SRC),'shard':SHARD,'shard_bytes':(SRC/SHARD).stat().st_size,'upstream_commit':'33b176e937ce6035ef39d4aab7414fd74a247424','before':memory_info(),'pairs':[],'GPU_initialized':False}
 assert out['host'].split('.')[0]=='a14r2n08';assert os.environ['SLURM_JOB_ID']=='910380'
 assert 0<out['shard_bytes']<=16*GIB
 assert out['before']['cgroups'],'no actual cgroup visibility: refuse preload'
 assert out['before']['available_bytes']>2*out['shard_bytes']+8*GIB
 stop=threading.Event();samples=[]
 def sample():
  while not stop.wait(.1):samples.append({'t':time.monotonic(),'memory':memory_info()})
 monitor=threading.Thread(target=sample,daemon=True);monitor.start();children=[];dp=None;sock=work/'run/d.sock'
 def read_pair(path,kind,roundid):
  ps=[];rows=[]
  try:
   for _ in range(2):
    p=subprocess.Popen([sys.executable,'-B',__file__,'reader',str(path),str(sock) if kind=='memfd' else 'none'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);ps.append(p);children.append(p)
   for p in ps:
    line=p.stdout.readline();assert line,(p.poll(),p.stderr.read());rows.append(json.loads(line))
   for row in rows:row['mapped_memory_held_two_readers']=mapped_memory(row['pid'],row['inode'])
   if kind=='memfd':
    out['active_leases_two_readers']=request(sock,{'command':'status'})['status']['active_leases'];assert out['active_leases_two_readers']==2
    try:request(sock,{'command':'stop','remove_view':True})
    except RuntimeError as e:out['stop_with_live_leases_refused']=str(e)
    else:raise AssertionError('stop accepted live lease')
   out['pairs'].append({'kind':kind,'round':roundid,'readers':rows})
  finally:
   for p in ps:
    if p.poll() is None:p.stdin.write('release\n');p.stdin.flush()
   for p in ps:
    so,se=p.communicate(timeout=30);assert p.returncode==0,(p.returncode,se)
 try:
  # Warmup intentionally reads the original once; there is no cache flush or cold claim.
  p=subprocess.run([sys.executable,'-B',__file__,'reader',str(SRC/SHARD),'none'],input='release\n',capture_output=True,text=True,timeout=300);assert p.returncode==0,p.stderr;out['unscored_original_warmup']=json.loads(p.stdout)
  log=(work/'daemon.log').open('x');t=time.perf_counter();dp=subprocess.Popen([sys.executable,'-B',__file__,'daemon',str(work)],stdout=log,stderr=subprocess.STDOUT);children.append(dp)
  deadline=time.monotonic()+300
  while not sock.exists():
   assert dp.poll() is None, (dp.returncode,(work/'daemon.log').read_text());assert time.monotonic()<deadline;time.sleep(.1)
  status=request(sock,{'command':'status'})['status'];out['daemon_ready_wall_seconds']=time.perf_counter()-t;out['daemon_status']=status
  assert status['file_count']==1 and status['total_bytes']==out['shard_bytes']
  manifest=request(sock,{'command':'manifest'})['manifest'];out['memfd_manifest']=manifest
  cached=work/'cached-view'/SHARD;assert cached.is_symlink();out['memfd_seals']=[]
  import fcntl
  with cached.open('rb') as f:out['memfd_seals']=fcntl.fcntl(f.fileno(),fcntl.F_GET_SEALS)
  overlay=work/'overlay';overlay.mkdir();count=0
  for p in sorted(SRC.rglob('*')):
   if p.is_file():
    rel=p.relative_to(SRC);dest=overlay/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(cached if rel.as_posix()==SHARD else p);count+=1
  out['private_overlay_file_links']=count;out['overlay_index_bytes_identity']=hashlib.sha256((overlay/'model.safetensors.index.json').read_bytes()).hexdigest()==hashlib.sha256((SRC/'model.safetensors.index.json').read_bytes()).hexdigest();assert out['overlay_index_bytes_identity']
  for roundid,order in enumerate([['original','memfd'],['memfd','original']],1):
   for kind in order:read_pair(SRC/SHARD if kind=='original' else overlay/SHARD,kind,roundid)
  reference=out['pairs'][0]['readers'][0]
  for pair in out['pairs']:
   for row in pair['readers']:
    for k in ['header_sha256','tensor_count','selected','page_checksum']:assert row[k]==reference[k],k
  out['identity_all_pairs']=True;assert request(sock,{'command':'status'})['status']['active_leases']==0
  request(sock,{'command':'stop','remove_view':True});dp.wait(timeout=15);assert dp.returncode==0
  out['clean_stop']={'daemon_exit':dp.returncode,'socket_removed':not sock.exists(),'pid_removed':not (work/'run/d.pid').exists(),'state_removed':not (work/'run/d.state.json').exists(),'cached_view_removed':not (work/'cached-view').exists(),'proc_memfds_gone':all(not Path(x['memfd']).exists() for x in manifest['files'])}
  assert all(v for k,v in out['clean_stop'].items() if k!='daemon_exit')
  for p in sorted(overlay.rglob('*'),key=lambda p:len(p.parts),reverse=True):
   if p.is_symlink():p.unlink()
   elif p.is_dir():p.rmdir()
  overlay.rmdir();out['overlay_removed']=True
  out['passed']=True
 except BaseException:
  out['passed']=False;out['failure']=traceback.format_exc();raise
 finally:
  for p in children:
   if p.poll() is None:p.terminate()
  for p in children:
   if p.poll() is None:
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
  stop.set();monitor.join(timeout=2);out['memory_samples']=samples;out['after']=memory_info();out['peak_parent_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
  (work/'result.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'passed':out['passed'],'result':str(work/'result.json'),'pairs':len(out['pairs'])}))

if __name__=='__main__':
 if sys.argv[1]=='reader':reader(Path(sys.argv[2]),sys.argv[3])
 elif sys.argv[1]=='daemon':sys.exit(daemon(sys.argv[2]))
 else:main(sys.argv[2])
