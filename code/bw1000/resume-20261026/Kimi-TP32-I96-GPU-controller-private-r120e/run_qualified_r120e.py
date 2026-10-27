"""Qualified entry. Delegates unchanged r120 math and one existing layer load."""
import sys,pathlib,json,os,datetime
from contract_r120e import gate,sha,CPU_SHA,DSO_SHA
m,g,c=gate(os.environ['R120E_MANIFEST_SHA256'],os.environ['R120E_GO_FILE'],container=True)
runtime=pathlib.Path(os.environ['R120_RUNTIME_MANIFEST']);assert sha(runtime)==os.environ['R120E_RUNTIME_SHA256'];q=json.loads(runtime.read_bytes())
assert q['explicit_component_GO']is True and q['CPU_receipt_SHA256']==CPU_SHA and q['library_SHA256']==DSO_SHA
assert q['root_GO_SHA256']==sha(os.environ['R120E_GO_FILE'])and q['publication_commit']==g['publication_commit']
assert q['fresh_before_age_at_submit_seconds']<=30 and q['master_addr']==q['fresh_before']['rows'][0]['ib0_ipv4'][0]
assert len(q['fresh_before']['rows'])==4 and all(x['passed']and not x['kfd_pids']and x['wildcard_port_bind_free']for x in q['fresh_before']['rows'])
assert all(0<=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(x['utc'])).total_seconds()<=120 for x in q['fresh_before']['rows'])
sys.path.insert(0,'/refactor/incoming/Kimi-TP32-I96-MoE-r120')
from run_component_r120 import run
output='/refactor/results/Kimi-TP32-I96-MoE-r120-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'-node'+os.environ['SLURM_PROCID']+'.json'
run(str(runtime),output)
p=pathlib.Path(output);a=json.loads(p.read_bytes());assert a['completed']and a['normal_CUDA_cleanup']
a.update(CPU_receipt_SHA256=CPU_SHA,library_SHA256=DSO_SHA,controller_manifest_SHA256=os.environ['R120E_MANIFEST_SHA256'],publication_commit=g['publication_commit'])
t=p.with_suffix('.qualified.tmp');t.write_text(json.dumps(a,indent=2));t.replace(p)
print('KIMI_TP32_I96_R120E_NORMAL',flush=True)
