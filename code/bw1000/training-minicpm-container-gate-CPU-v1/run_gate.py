"""CPU-only check of HOST FD and container kernel visibility; no SDK/Torch."""
import argparse,hashlib,json,os,socket,subprocess,time
from pathlib import Path
from snapshot import snapshot,gpu_holders
parser=argparse.ArgumentParser();parser.add_argument('--container',action='store_true');options=parser.parse_args()
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-container-gate-CPU-v1');start=time.monotonic()
if options.container:
 value=snapshot(True);assert value['full_fd_inventory_performed'] is False and value['holder_inventory_basis']=='kernel_KFD_proc_only'
 try:value['diagnostic_old_full_fd_scan_result']=gpu_holders();value['diagnostic_old_full_fd_scan_error']=None
 except PermissionError:value['diagnostic_old_full_fd_scan_result']=None;value['diagnostic_old_full_fd_scan_error']='PermissionError'
 print(json.dumps(value))
else:
 assert socket.gethostname().split('.')[0]=='f11r4n09' and os.environ['SLURM_JOB_ID']=='1087656'
 epoch=float(os.environ['GATE_CPU_EPOCH']);deadline=float(os.environ['GATE_CPU_DEADLINE']);assert deadline==epoch+150 and epoch<=time.time()<deadline
 before=snapshot(False);assert before['full_fd_inventory_performed'] and before['kfd_holder_count']==0
 shell='set -euo pipefail; source /opt/dtk/env.sh; export PYTHONDONTWRITEBYTECODE=1; exec python3 -B -u /stage-source/run_gate.py --container'
 command=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--rocm','--no-home','--pwd','/','--bind',str(W/'source')+':/stage-source:ro','--bind','/opt/hyhal:/opt/hyhal:ro','/public/home/aclsylqidf/hygon-inference-refactor-20261014/containers/dtk26-sandbox-r1','/bin/bash','--noprofile','--norc','-c',shell]
 p=subprocess.run(command,capture_output=True,timeout=min(90,deadline-time.time()-20));(W/'container.stdout').write_bytes(p.stdout);(W/'container.stderr').write_bytes(p.stderr)
 assert p.returncode==0
 inside=json.loads(p.stdout);after=snapshot(False)
 assert inside['kfd_holder_count']==0 and after['kfd_holder_count']==0 and after['full_fd_inventory_performed']
 assert before['pci']==inside['pci']==after['pci'] and sorted(inside['pci_to_kfd'])==before['pci'] and before['boot_id']==inside['boot_id']==after['boot_id']
 assert 0<=after['monotonic_ns']-before['monotonic_ns']<=45_000_000_000
 row=dict(passed=True,qualification='CPU-only visibility boundary. No device lease, SDK/Torch/model, native math or GPU qualification.',owner=dict(node='f11r4n09',step='1087656.'+os.environ['SLURM_STEP_ID'],PID=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])),before=before,container=inside,after=after,container_stderr_SHA256=hashlib.sha256(p.stderr).hexdigest(),container_returncode=p.returncode,explicit_program_SDK_calls=0,container_implicit_SDK_calls='UNKNOWN',wall_seconds=time.monotonic()-start,GPU_cards=0,Model_GO=False)
 with (W/'CPU-result.json').open('x') as f:json.dump(row,f,indent=2)
 print(json.dumps(row))
