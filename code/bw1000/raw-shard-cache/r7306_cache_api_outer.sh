#!/bin/bash
set -euo pipefail
ulimit -c 0
R=/public/home/aclsylqidf/hygon-inference-refactor-20261014
OWN=/work2/aclsylqidf-hygon-inference-r7302
python3 - <<'PY'
import pathlib,os,json,subprocess,socket
root=pathlib.Path('/work2/aclsylqidf-hygon-inference-r7302');root.mkdir(mode=0o700,exist_ok=True);assert root.stat().st_uid==os.getuid()and not root.is_symlink()
marker=root/'owner-r7303.json'
if marker.exists():assert json.loads(marker.read_text())['project']=='hygon-inference-refactor-20261014'
else:
 with marker.open('x')as f:json.dump({'project':'hygon-inference-refactor-20261014','CPU_single_shard_only':True},f)
st=os.statvfs(root);out={'node':socket.gethostname(),'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'owned_path':str(root),'statvfs_available_bytes':st.f_bavail*st.f_frsize,'scope':'Availablefilesystemspace not userquota or futurememoryheadroom','findmnt':{},'memory_cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'SLURM_CPUS_PER_TASK':os.environ.get('SLURM_CPUS_PER_TASK'),'SLURM_MEM_PER_NODE':os.environ.get('SLURM_MEM_PER_NODE')}
for p in ['/work2',str(root),'/public/home/aclsylqidf/shared/Kimi-K3']:out['findmnt'][p]=subprocess.check_output(['findmnt','-T',p,'-n','-o','SOURCE,FSTYPE,TARGET,OPTIONS'],text=True).strip()
out['MemAvailable']=[v for v in pathlib.Path('/proc/meminfo').read_text().splitlines()if v.startswith(('MemAvailable:','MemFree:'))]
source=pathlib.Path('/public/home/aclsylqidf/shared/Kimi-K3');sizes=[{'name':p.name,'bytes':p.stat().st_size}for p in sorted(source.glob('*.safetensors'))]
selected=source/'model-00011-of-000096.safetensors';before=selected.stat();assert before.st_size==16990916912 and before.st_size<=24*1024**3
out['shard_selection']={'all_shard_sizes':sizes,'selected':selected.name,'bytes':before.st_size,'source_metadata':{k:getattr(before,k)for k in ['st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns']}}
out['owned_directory_occupancy_bytes']=int(subprocess.check_output(['du','-sb',str(root)],text=True).split()[0])
out['required_free_bytes']=2*before.st_size+1024**3
out['filesystem_free_gate_passed']=out['statvfs_available_bytes']>=out['required_free_bytes']
available=int(next(v.split()[1]for v in pathlib.Path('/proc/meminfo').read_text().splitlines()if v.startswith('MemAvailable:')))*1024
out['host_MemAvailable_bytes']=available
out['physical_available_gate_passed']=available>=out['required_free_bytes']
out['memory_v1_ancestors']=[]
for line in pathlib.Path('/proc/self/cgroup').read_text().splitlines():
 parts=line.split(':',2)
 if len(parts)==3 and'memory'in parts[1].split(','):
  base=pathlib.Path('/sys/fs/cgroup/memory');cur=base/parts[2].lstrip('/')
  while cur!=base.parent:
   row={'path':str(cur)}
   for field in ['memory.usage_in_bytes','memory.limit_in_bytes','memory.max_usage_in_bytes','memory.failcnt','memory.stat']:
    p=cur/field
    if p.exists():row[field]=p.read_text()
   out['memory_v1_ancestors'].append(row)
   if cur==base:break
   cur=cur.parent
out['memory_scope']='Hostavailability and actual cgroupchargedusage/cache recorded independently; clean pagecache is not certified free RAM.64GB CPU step; no memfd/wholeweightresident.'
p=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')/('r7306-host-preflight-'+out['step']+'.json');assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps(out));assert out['filesystem_free_gate_passed']and out['physical_available_gate_passed'],'actualfree_below_2xshard_plus1GiB_NO_COPY'
PY
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
/public/software/apps/apptainer/1.3.4/bin/apptainer exec --home /public/home/aclsylqidf/hygon-inference-20261013/build-deps/root:/root \
 --bind "$R:/refactor" --bind "$OWN:$OWN" \
 --bind /public/home/aclsylqidf/shared/Kimi-K3:/public/home/aclsylqidf/shared/Kimi-K3:ro \
 "$R/containers/dtk26-sandbox-r1" python3 -B /refactor/incoming/r7306_cache_api_inside.py
