"""One CPU-only passive KFD/PID/43843 observation; never connects or kills."""
from pathlib import Path
import datetime,json,os,socket,time
node=socket.gethostname().split('.')[0];assert node in ['a02r4n12','a02r4n13'];started=time.time()
limit=float(os.environ.get('GLM_R404_POST_DEADLINE_UNIX','0'));expected=json.loads(os.environ.get('GLM_R404_EXPECTED_PIDS','{}')).get(node,[])
while True:
 kfd=Path('/sys/class/kfd/kfd/proc');assert kfd.is_dir();pids=sorted(int(p.name)for p in kfd.iterdir()if p.name.isdigit());matching=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit()or int(proc.name)==os.getpid():continue
  try:args=(proc/'cmdline').read_bytes().split(b'\0')
  except OSError:continue
  if any(b'/GLM-V-on-demand-Model-r404/owner/'in a and any(a.endswith(x)for x in [b'/model_entry_r290.py',b'/client_driver_r290.py',b'/client_crossworkload_r349.py'])for a in args):matching.append(int(proc.name))
 with socket.socket()as s:
  try:s.bind(('0.0.0.0',43843));available=True
  except OSError:available=False
 alive=[p for p in expected if Path('/proc/'+str(p)).exists()]
 if available or pids or matching or alive or time.time()>=min(limit,started+90):break
 time.sleep(.5) # same original cleanup deadline; no connection/restart
q=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),node=node,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],CPU_only=True,GPU_initialized=False,KFD_available=True,kfd_pids=pids,matching_component_processes=matching,expected_PIDs_still_exist=alive,ports_bindable={'43843':available},TCP_connections_made=0,passive_port_wait_s=time.time()-started)
print(json.dumps(q));assert not pids and not matching and not alive and available
