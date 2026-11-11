"""Bind the actual Slurm HIP device identity before running the pinned diagnostic."""
import ctypes,json,os,re,runpy,socket
from pathlib import Path
assert socket.gethostname().split('.')[0]=='f11r4n09' and os.environ['SLURM_JOB_ID']=='1087656'
hip=ctypes.CDLL('libamdhip64.so');count=ctypes.c_int();pci=ctypes.create_string_buffer(32)
assert hip.hipGetDeviceCount(ctypes.byref(count))==0 and count.value==1
assert hip.hipDeviceGetPCIBusId(pci,len(pci),0)==0
name=pci.value.decode().lower()
if re.fullmatch(r'[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',name):name='0000:'+name
assert re.fullmatch(r'[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',name)
stat=Path('/proc/self/stat').read_text().split(') ',1)[1].split()
with Path('/probe/assigned.json').open('x') as f:json.dump({'PCI':name,'visible_cards':count.value,'step':'1087656.'+os.environ['SLURM_STEP_ID'],'node':'f11r4n09','PID':os.getpid(),'startticks':int(stat[19])},f)
runpy.run_path('/probe/src/check_GPU.py',run_name='__main__')
