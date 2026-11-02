"""Prepare source-bound once-only launch; publication commit is supplied after push.

No jobs are started by this local generator. Remote invoke_once owns the keeper;
the published controller makes fresh five-node probes before creating any GPU step.
"""
from pathlib import Path
import argparse,ast,base64,hashlib,json

def main():
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);args=p.parse_args()
    assert len(args.publication)==40 and all(c in '0123456789abcdef' for c in args.publication)
    D=Path(__file__).resolve().parent;G=D.parents[2];L=G.parent
    mf=json.loads((D/'GPU_MANIFEST.json').read_bytes())
    h=hashlib.sha256((D/'GPU_MANIFEST.json').read_bytes()).hexdigest()
    assert mf['GPU_GO'] is True and mf['source_manifest_SHA256']==hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()
    assert mf['launcher_SHA256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan=json.loads((D/'controller/execution-plan.template.json').read_bytes())
    plan.update(execute_once=True,publication_commit=args.publication,
                controller_manifest_SHA256=mf['controller_manifest_SHA256'],
                target_manifest_SHA256=mf['target_manifest_SHA256'],
                CPU_receipt_SHA256=mf['controller_CPU_receipt_SHA256'],
                target_CPU_receipt_SHA256=mf['target_CPU_receipt_SHA256'],
                GPU_manifest_SHA256=h)
    script='''from pathlib import Path
import ast,base64,hashlib,json,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-I192-Model-pair-r389'
MF=__MF__;MFSHA=__MFSHA__;PLAN=__PLAN__
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(D/'SOURCE_MANIFEST.json')==MF['source_manifest_SHA256']
source=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
for name,digest in source['files_SHA256'].items():assert sha(D/name)==digest,name
for role in ['target','controller']:
 assert sha(D/role/'SOURCE_MANIFEST.json')==MF[role+'_manifest_SHA256']
 receipt=R/('candidates/Kimi-I192-Model-pair-r389/'+role+'/CPU-result.json')
 assert sha(receipt)==MF[role+'_CPU_receipt_SHA256']
 q=json.loads(receipt.read_bytes());assert q['passed'] and q['completed'] and q['GPU']==0 and not q['Native_dlopen']
assert sha(R/'candidates/Kimi-long-capture-link-CPU-r327/libKimi_long_capture_r327.so')==PLAN['Native_SHA256']
run=PLAN['run']
for name in ['Kimi-long-pair-r347-'+run+'-intent.json','Kimi-long-pair-r347-'+run+'-watch.json']:
 assert not (R/'results'/name).exists(),'Original intent/watch exists; follow it, never replay'
for prefix in ['Kimi-long-draft-start-r347-','Kimi-long-draft-listening-r347-','Kimi-long-pair-cancel-r347-']:
 assert not (R/'control'/(prefix+run+'.json')).exists(),'Original marker exists; reconcile'
# Immutable claim data publication; this never overwrites executed runtime source.
mp=D/'GPU_MANIFEST.json'
with mp.open('x')as f:json.dump(MF,f,indent=2);f.write('\\n')
assert sha(mp)==MFSHA
path=R/'control'/('Kimi-I192-Model-r389-'+run+'-plan.json')
with path.open('x')as f:json.dump(PLAN,f,indent=2);f.write('\\n')
proc=subprocess.run(['python3','-B',str(D/'controller/invoke_once.py'),'--plan',str(path)],text=True,capture_output=True,timeout=150)
if proc.returncode:raise RuntimeError(json.dumps(dict(rc=proc.returncode,stdout=proc.stdout[-3000:],stderr=proc.stderr[-3000:])))
result=json.loads(proc.stdout);result.update(publication_commit=PLAN['publication_commit'],GPU_manifest_SHA256=MFSHA,plan_path=str(path),plan_SHA256=sha(path))
print(json.dumps(result))
'''
    script=script.replace('__MF__',repr(mf)).replace('__MFSHA__',repr(h)).replace('__PLAN__',repr(plan))
    ast.parse(script)
    (L/'remote389-launch-once.py').write_text(script,encoding='utf8',newline='\n')
    print(json.dumps(dict(launch_helper_AST=True,GPU_manifest_SHA256=h,publication_commit=args.publication,submitted=False)))

if __name__=='__main__':main()
