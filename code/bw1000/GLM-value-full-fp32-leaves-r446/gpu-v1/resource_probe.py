"""Assigned host metadata only, no GPU runtime initialization."""
from pathlib import Path
import os,socket,json,sys
assert socket.gethostname().split('.')[0]=='a02r4n07';expected=json.loads(sys.argv[1]);k=Path('/sys/class/kfd/kfd/proc');assert k.is_dir();kfd=sorted(int(p.name)for p in k.iterdir()if p.name.isdigit());owned=[]
PCI=sorted({p.resolve().name for p in Path('/sys/class/drm').glob('renderD*/device')});assert len(PCI)==8
for p in Path('/proc').iterdir():
 if not p.name.isdigit()or int(p.name)==os.getpid():continue
 try:cmd=(p/'cmdline').read_bytes()
 except(OSError,PermissionError):continue
 if b'/GLM-value-full-fp32-leaves-r446/gpu-v1/GPU_run.py'in cmd:owned.append(int(p.name))
alive=[p for p in expected if Path('/proc',str(p)).exists()]
q=dict(CPU_only=True,GPU_initialized=False,node=socket.gethostname().split('.')[0],actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],KFD_pids=kfd,matching_GPU_driver_PIDs=owned,expected_owner_PIDs=expected,expected_PIDs_still_exist=alive,physical_PCI=PCI,PCI_scope='DRM readonlyinventory;actualCUDAindexmappingvalidatedintheoriginalGPUentry',fixed_application_ports=[],port_scope='No fixed component HTTP/control ports; actual knownPIDs gone proves ownedFDclosure, arbitrary TIME_WAIT bindability not claimed',passed=not kfd and not owned and not alive)
print(json.dumps(q));assert q['passed']
