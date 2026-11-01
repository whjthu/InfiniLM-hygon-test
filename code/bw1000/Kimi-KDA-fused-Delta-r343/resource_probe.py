"""CPU-only two-node resource probe. No CUDA, Torch or Runtime imports."""
from pathlib import Path
import os,socket,json,fcntl,struct,datetime
node=socket.gethostname().split('.')[0];p=Path('/sys/class/kfd/kfd/proc');assert p.exists()
pids=sorted(int(v.name)for v in p.iterdir()if v.name.isdigit())
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
ip=socket.inet_ntoa(fcntl.ioctl(s.fileno(),0x8915,struct.pack('256s',b'ib0'))[20:24]);s.close()
with socket.socket()as listener:listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);listener.bind(('',30040))
print(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':node,
 'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'CPU_only':True,
 'GPU_runtime_initialized':False,'kfd_pids':pids,'KFD_empty':not pids,'control30040_bindable':True,'ib0_ipv4':ip}))
