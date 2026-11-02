"""Generate a single source-bound keeper invoke after finalpublication/rootread."""
from pathlib import Path
import argparse,ast,base64,hashlib,json

def main():
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);a=p.parse_args();assert len(a.publication)==40 and all(x in '0123456789abcdef' for x in a.publication)
    D=Path(__file__).resolve().parent;L=D.parents[2].parent
    raw=(D/'GPU_MANIFEST.json').read_bytes();m=json.loads(raw);h=hashlib.sha256(raw).hexdigest()
    assert m['GPU_GO']is True and m['Model_GO']is False and m['launcher_SHA256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload={name:base64.b64encode((D/'controller'/name).read_bytes()).decode() for name in m['controller_files_SHA256']}
    script='''from pathlib import Path
import ast,base64,hashlib,json,os,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/GLM-C1-PP2-constructor-r395';MF=__MF__;DIGEST=__DIGEST__;PUBLICATION=__PUBLICATION__
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(D/'SOURCE_MANIFEST.json')==MF['source_manifest_SHA256']
for name,h in json.loads((D/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(D/name)==h,name
receipt=R/'candidates/GLM-C1-PP2-constructor-r395/CPU-result.json';assert sha(receipt)==MF['CPU_receipt_SHA256']
cpu=json.loads(receipt.read_bytes());assert cpu['completed']and cpu['passed']and cpu['GPU']==0 and not cpu['Native_dlopen']
label='GLM-C1-PP2-constructor-r395';run='constructor-20261102-a'
rows=subprocess.check_output(['sacct','-S','2026-10-31','-j','910376,910407','-n','-P','-o','JobIDRaw,JobName%160,State'],text=True,timeout=25)
names={line.split('|')[1]for line in rows.splitlines()if '|'in line}
assert label+'-'+run not in names and label+'-keeper-'+run not in names,'Original label exists;reconcile,no replay'
watch=R/'results'/(label+'-'+run+'-watch.json');intent=R/'results'/(label+'-'+run+'-intent.json')
assert not watch.exists()and not intent.exists()and not(D/'controller').exists()
(D/'controller').mkdir()
for name,value in __PAYLOAD__.items():(D/'controller'/name).write_bytes(base64.b64decode(value))
with(D/'GPU_MANIFEST.json').open('x')as f:json.dump(MF,f,indent=2);f.write('\\n')
assert sha(D/'GPU_MANIFEST.json')==DIGEST
for name,h in MF['controller_files_SHA256'].items():assert sha(D/'controller'/name)==h,name
with intent.open('x')as f:json.dump(dict(publication_commit=PUBLICATION,GPU_manifest_SHA256=DIGEST,CPU_keeper_only=True,GPU_submitted=False,Model_GO=False,retry=False),f)
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
argv=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a14r2n20','--gres=none','--mem=2048M','-t','00:10:00','--job-name='+label+'-keeper-'+run,'--output='+str(R/'logs'/(label+'-keeper-'+run+'-%J.log')),'python3','-B',str(D/'controller/controller.py'),'--publication-commit',PUBLICATION,'--manifest-sha',DIGEST]
with(R/'logs'/(label+'-keeper-'+run+'-client.log')).open('x')as log:p=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(CPU_keeper_client_PID=p.pid,publication_commit=PUBLICATION,GPU_manifest_SHA256=DIGEST,watch=str(watch),GPU_submitted_by_invoker=False,Model_GO=False)))
'''
    script=script.replace('__MF__',repr(m)).replace('__DIGEST__',repr(h)).replace('__PUBLICATION__',repr(a.publication)).replace('__PAYLOAD__',repr(payload));ast.parse(script)
    (L/'remote395-launch-once.py').write_text(script,encoding='utf8',newline='\n')
    print(json.dumps(dict(helper_AST=True,GPU_manifest_SHA256=h,publication_commit=a.publication,submitted=False,Model_GO=False)))

if __name__=='__main__':main()
