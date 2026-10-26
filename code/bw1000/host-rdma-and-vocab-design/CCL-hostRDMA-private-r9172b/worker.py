import pathlib,json,subprocess,socket,os,time,fcntl,struct,hashlib,resource,traceback
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');node=socket.gethostname();rank=int(os.environ['SLURM_PROCID']);assert node==['a02r3n17','a02r3n18'][rank];step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'results'/('CCL-hostRDMA-r9172b-'+step);out.mkdir(exist_ok=True);owned=[]
def atom(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def ip():
 with socket.socket(socket.AF_INET,socket.SOCK_DGRAM)as s:return socket.inet_ntoa(fcntl.ioctl(s.fileno(),0x8915,struct.pack('256s',b'ib0'))[20:24])
def cpulist(s):
 a=[]
 for part in s.strip().split(','):
  if '-'in part:x,y=map(int,part.split('-'));a+=list(range(x,y+1))
  elif part:a.append(int(part))
 return a
allowed=sorted(os.sched_getaffinity(0));railmeta=[];pool=[]
for i in range(4):
 p=pathlib.Path('/sys/class/infiniband')/('shca_'+str(i));numa=int((p/'device/numa_node').read_text());near=cpulist((pathlib.Path('/sys/devices/system/node')/('node'+str(numa))/'cpulist').read_text())if numa>=0 else allowed;choices=[c for c in near if c in allowed];cpu=next((c for c in choices if c not in pool),choices[0]if choices else allowed[i%len(allowed)]);pool.append(cpu);railmeta.append({'device':'shca_'+str(i),'numa':numa,'CPU':cpu,'CPU_is_NUMA_local':cpu in choices,'pci':str((p/'device').resolve())})
cpus=list(dict.fromkeys(pool));cpus+=[c for c in allowed if c not in cpus][:8-len(cpus)];os.sched_setaffinity(0,cpus);assert len(os.sched_getaffinity(0))<=8
q={'node':node,'rank':rank,'actual_step':step,'CPU_only':True,'GPU_context_created':False,'host_memory_only':True,'use_rocm_supported_by_installed_tools':False,'CPU_affinity':sorted(os.sched_getaffinity(0)),'rail_placement':railmeta,'ib0_IPv4':ip(),'RLIMIT_MEMLOCK':list(resource.getrlimit(resource.RLIMIT_MEMLOCK)),'cases':[],'completed':False};report=out/('node'+str(rank)+'.json')
def save():q['heartbeat_UTC']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat();atom(report,q)
def counters():
 result={}
 for i in range(4):
  p=pathlib.Path('/sys/class/infiniband')/('shca_'+str(i))/'ports/1';d={}
  for n in ['port_xmit_data','port_rcv_data','port_xmit_packets','port_rcv_packets','port_xmit_wait','port_rcv_errors','port_xmit_discards','symbol_error','link_downed']:
   f=p/'counters'/n
   if f.exists():d[n]=int(f.read_text().strip())
  result['shca_'+str(i)]=d
 return result

def send(d):f.write(json.dumps(d).encode()+b'\n');f.flush()
def recv():
 raw=f.readline();assert raw,'control closed';d=json.loads(raw);assert d['step']==step;return d

def phase(index,case,head):
 tool=case['tool'];rails=case['rails'];warm=case.get('warm',False)
 if rank==1:
  ready=recv();assert ready['event']=='servers_ready'and ready['index']==index
 before=counters();ps=[];t=time.perf_counter()
 for i in rails:
  port=23000+index*8+i;a=[tool,'-d','shca_'+str(i),'-i','1','-s','1048576','-q','1','-t','128','--report_gbits','-F','-N']
  if warm:a+=['-n','1000']
  else:a+=['-D','8','-f','1']
  a+=['-p',str(port)]
  if rank==1:a+=[head]
  op=out/(case['label']+'-rank'+str(rank)+'-rail'+str(i)+'.stdout');ep=out/(case['label']+'-rank'+str(rank)+'-rail'+str(i)+'.stderr');fo=op.open('x');fe=ep.open('x');cpu=railmeta[i]['CPU'];p=subprocess.Popen(a,stdout=fo,stderr=fe,preexec_fn=lambda c=cpu:os.sched_setaffinity(0,[c]));owned.append(p);ps.append((p,fo,fe,a,op,ep,i))
 if rank==0:
  time.sleep(.7)
  assert all(p.poll()is None for p,*rest in ps),'RDMAserver exited before client';send({'step':step,'event':'servers_ready','index':index});remote=recv();assert remote['index']==index and remote['event']=='client_done'
 results=[]
 for p,fo,fe,a,op,ep,i in ps:
  try:rc=p.wait(timeout=45)
  except subprocess.TimeoutExpired:p.terminate();raise
  fo.close();fe.close();raw=op.read_text(errors='replace');err=ep.read_text(errors='replace');rows=[]
  for s in raw.splitlines():
   tok=s.split()
   if len(tok)>=5 and tok[0]=='1048576':
    try:rows.append({'bytes':int(tok[0]),'iterations':int(tok[1]),'peak_Gbit_s':float(tok[2]),'average_Gbit_s':float(tok[3]),'Mmsgs_s':float(tok[4])})
    except ValueError:pass
  results.append({'rail':i,'args':a,'returncode':rc,'stdout_path':str(op),'stderr_path':str(ep),'stdout_SHA256':hashlib.sha256(op.read_bytes()).hexdigest(),'stdout':raw,'stderr':err,'rows':rows});assert rc==0 and rows,(case,i,rc,raw,err)
 after=counters();delta={dev:{k:val-before[dev][k]for k,val in vals.items()}for dev,vals in after.items()};record={'index':index,'label':case['label'],'tool':tool,'rails':rails,'warmup_excluded':warm,'buffer_bytes':1048576,'QP_per_rail':1,'duration_seconds':None if warm else 8,'margin_seconds':None if warm else 1,'wall_seconds':time.perf_counter()-t,'results':results,'counter_before':before,'counter_after':after,'counter_delta':delta,'reported_aggregate_average_Gbit_s':sum(x['rows'][-1]['average_Gbit_s']for x in results)};q['cases'].append(record);save()
 if rank==1:send({'step':step,'event':'client_done','index':index})
 if rank==0:send({'step':step,'event':'phase_done','index':index})
 else:assert recv()['event']=='phase_done'

save()
try:
 if rank==0:
  listener=socket.socket();listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);listener.bind((q['ib0_IPv4'],22173));listener.listen(1);listener.settimeout(60);assert q['ib0_IPv4']==os.environ['RDMA_HEAD_R9172B'];conn,peer=listener.accept();conn.settimeout(90);f=conn.makefile('rwb');assert recv()['event']=='join';send({'step':step,'event':'ready'});head=q['ib0_IPv4']
 else:
  deadline=time.monotonic()+60;head=os.environ['RDMA_HEAD_R9172B']
  while True:
   try:conn=socket.create_connection((head,22173),timeout=3);break
   except OSError:
    assert time.monotonic()<deadline;time.sleep(.2)
  conn.settimeout(90);f=conn.makefile('rwb');send({'step':step,'event':'join'});assert recv()['event']=='ready'
 cases=[{'label':'warm-write-4rail','tool':'ib_write_bw','rails':[0,1,2,3],'warm':True}]+[{'label':'write-single-rail'+str(i),'tool':'ib_write_bw','rails':[i]}for i in range(4)]+[{'label':'write-4rail','tool':'ib_write_bw','rails':[0,1,2,3]},{'label':'read-single-rail0','tool':'ib_read_bw','rails':[0]},{'label':'read-4rail','tool':'ib_read_bw','rails':[0,1,2,3]},{'label':'send-single-rail0','tool':'ib_send_bw','rails':[0]},{'label':'send-4rail','tool':'ib_send_bw','rails':[0,1,2,3]}]
 for index,case in enumerate(cases):
  if rank==0:send({'step':step,'event':'next','index':index})
  else:assert recv()['index']==index
  phase(index,case,head)
 q['completed']=True;q['passed']=True;save();print('CCL_HOST_RDMA_R9172B_NORMAL '+node,flush=True)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc(),passed=False);save();raise
finally:
 for p in owned:
  if p.poll()is None:
   p.terminate()
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
print(json.dumps({'node':node,'actual_step':step,'report':str(report),'completed':q['completed'],'passed':q.get('passed',False)}),flush=True)
