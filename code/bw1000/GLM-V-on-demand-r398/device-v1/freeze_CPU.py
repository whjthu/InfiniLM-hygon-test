from pathlib import Path
import json,hashlib,ast
from check_CPU import run
D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 for p in D.glob('*.py'):ast.parse(p.read_text());compile(p.read_bytes(),str(p),'exec')
 for n in ['CPU_inside.sh','GPU_inside.sh']:assert b'\r'not in(D/n).read_bytes()
 q=run();assert q['passed'];q['scope']='Local real CPU child + actualcontroller/loader/schedule bodies with explicitly recording Slurm/Native leaves; no device/math qualification'
 q['initial_unpublished_local_check_failure']='Windows recording cleanup lacked SIGKILL/killpg; test leaves now portable and real CPU child waited before synthetic accounting. No Slurm/Native/GPU or source789 changed.'
 (D/'local-CPU-result.json').write_text(json.dumps(q,indent=2)+'\n',encoding='utf8')
 files=['attention_component_r398.cpp','bf16_metrics_r398.hpp','CPU_metrics_contract.cpp','consumer.py','checkpoint_loader.py','GPU_run.py','controller.py','GPU_inside.sh','resource_probe.py','CONTROLLER_DERIVATION.json','check_CPU.py','CPU_build.py','CPU_inside.sh','PLAN.json','freeze_CPU.py','local-CPU-result.json']
 m=dict(schema=1,files_SHA256={n:sha(D/n)for n in files},CPU_claim=json.loads((D/'PLAN.json').read_bytes())['CPU_claim'],parent_CPU_receipt_SHA256='399f26b7668f3fa21a22ab81a041cb5b1b1835aad50a3d18fa6a6ae8196f311b',GPU_GO=False,Model_GO=False)
 p=D/'CPU_SOURCE_MANIFEST.json';p.write_text(json.dumps(m,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(SHA256=sha(p),members=len(files),actual_local_recording=q['passed'])))
if __name__=='__main__':main()
