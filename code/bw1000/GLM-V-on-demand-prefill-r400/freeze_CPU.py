from pathlib import Path
import ast,json,hashlib
from check_CPU import run
D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 for p in D.glob('*.py'):ast.parse(p.read_text());compile(p.read_bytes(),str(p),'exec')
 for p in D.glob('*.sh'):assert b'\r'not in p.read_bytes()
 q=run();assert q['passed'];assert q['schedule']['query_calls']==6 and q['schedule']['seed_calls']==384 and q['schedule']['diagnostic_V_calls']==1 and q['schedule']['inspect_failure_primary_persisted']
 q['scope']='Local actualconsumer/controller/loader bodies and realfile-before-failinginspect with recordingNative/Slurm leaves. ActualsharedCXXchunkloop and changedconsumercompile await LinuxCPU; not GPU/math.'
 q['unpublished_local_check_failure']='Clonedclose-failure recording leaf initially renamed compatibility CABI to r400; restored actualretained r398 methodnames. No Slurm/Native/GPU/oldsource changed.'
 (D/'local-CPU-result.json').write_text(json.dumps(q,indent=2)+'\n',encoding='utf8')
 files=sorted(p.name for p in D.iterdir()if p.is_file()and p.name!='CPU_SOURCE_MANIFEST.json')
 m=dict(schema=1,files_SHA256={n:sha(D/n)for n in files},CPU_claim=json.loads((D/'PLAN.json').read_bytes())['CPU_claim'],parent_CPU='910407.798',parent_CPU_receipt_SHA256='92103aba80235b27780abf942a43ad2e4a3cdc80119143297285e768c89d9f7c',GPU_GO=False,Model_GO=False)
 p=D/'CPU_SOURCE_MANIFEST.json';p.write_text(json.dumps(m,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(SHA256=sha(p),members=len(files),queries=6,seeds=384,GPU_GO=False)))
if __name__=='__main__':main()
