"""Follow only original367 CPU label; never submit or dlopen."""
from pathlib import Path
import json,subprocess,hashlib,datetime
R=Path.home()/'hygon-inference-refactor-20261014';q=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU=False)
raw=subprocess.check_output(['sacct','-S','2026-11-01','-j','910407','-n','-P','-o','JobIDRaw,JobName%160,State,ExitCode,Elapsed,NodeList%100'],text=True,timeout=25)
q['accounting']=[line.split('|')for line in raw.splitlines()if line.strip()and line.split('|')[1]=='GLM-r367-M1-gate-CPU'];assert len(q['accounting'])<=1
p=R/'candidates/GLM-smallM-FFN-r367/CPU-build.json'
if p.exists():
 try:
  a=json.loads(p.read_bytes());q['CPU']={k:a.get(k)for k in ['completed','CPU_only','GPU_initialized','dlopen','weights_opened','producer','CPU_contract','source_manifest_SHA256','parent_CPU_receipt_SHA256','parent_native_SHA256','library','library_SHA256','source_SHA256','object_SHA256','replaced_object','reused_objects_SHA256','actual_dynamic_dependency_SHA256','all_parent_exports_retained','Frame_ABI_unchanged','default_OFF','error','traceback','elapsed_seconds']};q['CPU_receipt_path']=str(p);q['CPU_receipt_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
 except json.JSONDecodeError:q['receipt_partial_JSON']=True
q['log_tails']={str(p):p.read_text(errors='replace')[-2500:]for p in (R/'logs').glob('GLM-r367-M1-gate-CPU*.log')}
print(json.dumps(q))
