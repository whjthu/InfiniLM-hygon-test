"""Two assigned nodes, CPU-only passive PID/KFD/PCI/port/IP inventory."""
from pathlib import Path
import fcntl,json,os,socket,struct,sys,time
node=socket.gethostname().split('.')[0];assert node in['a02r3n03','a02r3n07']
expected=json.loads(sys.argv[1]).get(node,[]);started=time.time();until=min(started+90,float(os.environ.get('KIMI_R440_POST_DEADLINE',str(started))))
PCI=sorted({p.resolve().name for p in Path('/sys/class/drm').glob('renderD*/device')});assert len(PCI)==8
addresses={}
with socket.socket(socket.AF_INET,socket.SOCK_DGRAM)as sock:
 for p in sorted(Path('/sys/class/net').iterdir()):
  if not p.name.startswith('ib'):continue
  try:addresses[p.name]=socket.inet_ntoa(fcntl.ioctl(sock.fileno(),0x8915,struct.pack('256s',p.name.encode()))[20:24])
  except OSError:pass
while True:
 k=Path('/sys/class/kfd/kfd/proc');assert k.is_dir();kfd=sorted(int(p.name)for p in k.iterdir()if p.name.isdigit());matching=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit()or int(p.name)==os.getpid():continue
  try:raw=(p/'cmdline').read_bytes()
  except OSError:continue
  if b'/Kimi-lowrank-TP-rowshard-r440/source-span-v2/gpu-v5/GPU_run.py'in raw:matching.append(int(p.name))
 alive=[v for v in expected if Path('/proc',str(v)).exists()]
 with socket.socket()as s:
  try:s.bind(('0.0.0.0',40917));port=True
  except OSError:port=False
 if port or kfd or matching or alive or time.time()>=until:break
 time.sleep(.5)
q=dict(CPU_only=True,GPU_initialized=False,node=node,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],KFD_pids=kfd,matching_GPU_driver_PIDs=matching,expected_PIDs_still_exist=alive,physical_PCI=PCI,ib_ipv4=addresses,passive_master_port40917_bindable=port,passive_port_wait_s=time.time()-started,TCP_connections=0,passed=not kfd and not matching and not alive and port)
print(json.dumps(q));assert q['passed']
