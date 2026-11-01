from pathlib import Path
import json
D=Path(__file__).resolve().parent;L=D.parents[3]
g=json.loads((D/'execution-GO.template.json').read_bytes());g['publication_commit']='998920b48dff109dace942e79eed435e4ad0f56f'
script="""from pathlib import Path
import json,hashlib,subprocess,sys
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/GLM-two-cohort-Model-r355/controller';g=json.loads(GO_TEXT)
sys.path.insert(0,str(D));from controller import verify
verify(R,D,g)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=R/'candidates/GLM-PP4-registry-repair-r184b/CPU-build.json';assert sha(receipt)=='6aa62f788aad6080027d665f30d7c9824c07cb1866de905397fef15c56b26529';q=json.loads(receipt.read_bytes())
def host(p):return R/p[len('/refactor/'):]if p.startswith('/refactor/')else Path(p)
assert sha(host(q['library']))==g['Native_SHA256']==q['library_SHA256']
bound={}
for p,h in q['actual_binary_SHA256'].items():
 if p.startswith('/refactor/'):
  assert sha(host(p))==h,p;bound[p]=h
card=json.loads((D.parent/'owner/FIXTURE_CARD.json').read_bytes());fixtures=[]
for f in card['fixtures']:
 raw=Path(f['request_external_path']);ids=host(f['current_prepared_prompt_IDs'])
 assert sha(raw)==f['request_SHA256']and sha(ids)==f['current_prepared_private_IDs_file_SHA256']
 fixtures.append({'id':f['id'],'request_SHA256':sha(raw),'IDs_file_SHA256':sha(ids),'tokens':f['current_prepared_prompt_tokens']})
check={'source_verified':True,'Native_SHA256':g['Native_SHA256'],'actual_binary_pins':bound,'fixtures':fixtures,'publication_commit':g['publication_commit'],'GPU_started_by_precheck':False}
pre=R/'results'/('GLM-r355-prelaunch-'+g['run']+'.json')
with pre.open('x')as f:json.dump(check,f,indent=2)
go=R/'control'/('GLM-r355-exec-'+g['run']+'.json')
with go.open('x')as f:json.dump(g,f,indent=2)
p=subprocess.run(['python3','-B',str(D/'invoke.py'),'--execution-go',str(go)],text=True,capture_output=True,timeout=65)
assert p.returncode==0,p.stderr
print(json.dumps({'prelaunch':check,'invoke':json.loads(p.stdout),'stderr':p.stderr[-1000:]}))
""".replace('GO_TEXT',repr(json.dumps(g)))
(L/'remote355-launch.py').write_text(script,encoding='utf8');print(json.dumps({'helper':'remote355-launch.py','run':g['run']}))
