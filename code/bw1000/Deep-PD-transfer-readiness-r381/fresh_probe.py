"""CPU-only bounded node inventory; no Torch, CUDA, model, network transfer or listener."""
from pathlib import Path
import os,socket,json,subprocess,shutil,resource,datetime

def text(p):
 try:return Path(p).read_text().strip()
 except (OSError,PermissionError):return None
def command(args,limit=24000):
 try:
  p=subprocess.run(args,capture_output=True,text=True,timeout=15)
  return dict(args=args,returncode=p.returncode,stdout=p.stdout[:limit],stderr=p.stderr[:4000])
 except (OSError,subprocess.TimeoutExpired)as e:return dict(args=args,error=repr(e))

q=dict(schema=1,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),node=socket.gethostname(),actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],CPU_only=True,GPU_initialized=False,model_loaded=False,transfer_started=False,CPU_affinity=sorted(os.sched_getaffinity(0)),memlock=list(resource.getrlimit(resource.RLIMIT_MEMLOCK)))
k=Path('/sys/class/kfd/kfd/proc');q['KFD_available']=k.exists();q['kfd_pids']=sorted(int(p.name)for p in k.iterdir()if p.name.isdigit())if k.exists()else None
q['GPU_compute_processes']=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if int(p.name)in(q['kfd_pids']or[])or any(a.endswith(('.py','.so'))and any(t in a.lower()for t in ['model_entry','owner','worker','gpu_driver','native'])for a in args):
   q['GPU_compute_processes'].append(dict(PID=int(p.name),argv=args[:24]))
 except (OSError,PermissionError):pass
q['network']={name:command(args)for name,args in [('addresses',['ip','-j','addr','show']),('routes',['ip','-j','route','show']),('RDMA_links',['rdma','link','show']),('ibstat',['ibstat']),('ibv_devices',['ibv_devices']),('ibv_devinfo',['ibv_devinfo','-v'])]}
q['RDMA_sysfs']=[]
for dev in sorted(Path('/sys/class/infiniband').glob('*')):
 row=dict(name=dev.name,PCI=str((dev/'device').resolve()),numa=text(dev/'device/numa_node'),ports=[])
 for port in sorted((dev/'ports').glob('*')):
  row['ports'].append(dict(port=port.name,**{n:text(port/n)for n in ['state','phys_state','rate','link_layer','lid','sm_lid']},netdevices=[p.name for p in (dev/'device/net').glob('*')]))
 q['RDMA_sysfs'].append(row)
q['GPU_PCI_sysfs']=[]
for p in Path('/sys/bus/pci/devices').glob('*'):
 if text(p/'vendor')=='0x1d94' or (p/'driver').exists()and (p/'driver').resolve().name=='amdgpu':
  q['GPU_PCI_sysfs'].append(dict(PCI=p.name,vendor=text(p/'vendor'),device=text(p/'device'),numa=text(p/'numa_node'),driver=(p/'driver').resolve().name if (p/'driver').exists()else None))
q['tools']={name:shutil.which(name)for name in ['ib_write_bw','ib_read_bw','ib_send_bw','ucx_info','mpirun','hipcc','rocminfo','rdma','ip']}
if q['tools']['ib_write_bw']:
 h=command([q['tools']['ib_write_bw'],'--help']);q['perftest_help']=h
 full=h.get('stdout','')+h.get('stderr','');q['perftest_ROCm_option_advertised']=any(s in full.lower()for s in ['--use_rocm','--use_hip','rocm memory','hip memory'])
if q['tools']['ucx_info']:q['UCX_devices']=command([q['tools']['ucx_info'],'-d'])
q['libraries']=command(['sh','-c',"ldconfig -p 2>/dev/null | grep -E 'ibverbs|rdmacm|rccl|ucp|uct|hsa-runtime|amdhip'"],12000)
q['candidate_transfer_ports_bindable']={}
for port in range(49181,49189):
 s=socket.socket()
 try:s.bind(('',port));q['candidate_transfer_ports_bindable'][str(port)]=True
 except OSError:q['candidate_transfer_ports_bindable'][str(port)]=False
 finally:s.close()
print(json.dumps(q))
