"""Same host before-KFD admission also reads actual cgroup headroom; no GPU init."""
import pathlib,subprocess,json,sys
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';sys.path.insert(0,str(r/'incoming'))
from kimi_MLP_layer0_memory_r8648 import read_memory
z=subprocess.run([sys.executable,str(r/'incoming/Kimi_Q0_idle_r8066.py')],capture_output=True,text=True,timeout=30);assert z.returncode==0,z.stderr
q=json.loads(z.stdout);m=read_memory('same_host_before_KFD_layer0');headroom=m['job_ancestor_current_headroom_bytes'];available=m['host_meminfo'].get('MemAvailable_bytes');required=32*1024**3
q.update(memory=m,memory_required_bytes=required,memory_passed=headroom is not None and available is not None and min(headroom,available)>=required,memory_scope='Actual currentjobancestor limit-usage and hostMemAvailable, same node, before uniqueGPU Popen. No cache eviction.')
print(json.dumps(q))
