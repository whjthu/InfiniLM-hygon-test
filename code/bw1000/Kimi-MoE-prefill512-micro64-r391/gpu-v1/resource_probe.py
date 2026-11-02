from pathlib import Path
import fcntl,json,os,socket,struct,sys
nodes=['a02r3n03','a02r3n07'];node=socket.gethostname().split('.')[0];assert node in nodes;mapping=json.loads(sys.argv[1]);expected=mapping.get(node,[]);k=Path('/sys/class/kfd/kfd/proc');assert k.is_dir();kfd=sorted(int(p.name)for p in k.iterdir()if p.name.isdigit());matching=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit()or int(p.name)==os.getpid():continue
 try:cmd=(p/'cmdline').read_bytes()
 except(OSError,PermissionError):continue
 if b'/Kimi-MoE-prefill512-micro64-r391-v2/GPU_run.py'in cmd:matching.append(int(p.name))
alive=[p for p in expected if Path('/proc',str(p)).exists()];ports=[];blocked=[];listeners=[];timewait=0
for file in ['/proc/net/tcp','/proc/net/tcp6']:
 for line in Path(file).read_text().splitlines()[1:]:
  fields=line.split();port=int(fields[1].split(':')[1],16)
  if 48091<=port<48107:
   if fields[3]=='0A':listeners.append(dict(file=file,port=port))
   if fields[3]=='06':timewait+=1
for port in range(48091,48107):
 try:
  with socket.socket()as s:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('',port));ports.append(port)
 except OSError as e:blocked.append(dict(port=port,error=str(e)))
with socket.socket(socket.AF_INET,socket.SOCK_DGRAM)as s:ip=socket.inet_ntoa(fcntl.ioctl(s.fileno(),0x8915,struct.pack('256s',b'ib0'))[20:24])
q=dict(CPU_only=True,GPU_initialized=False,node=node,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],KFD_pids=kfd,matching_GPU_driver_PIDs=matching,expected_owner_PIDs=expected,expected_PIDs_still_exist=alive,ib0_IPv4=ip,bindable_ports=ports,port_bind_diagnostics=blocked,listeners=listeners,TIME_WAIT_count=timewait,TIME_WAIT_not_listener=True,post=bool(mapping),passed=not kfd and not matching and not alive and not listeners and(bool(mapping)or not blocked))
print(json.dumps(q));assert q['passed']
