"""Read-only AST and SHA checks; writes only local launch manifests, no execution."""
from pathlib import Path
import ast, hashlib, json
D=Path(__file__).resolve().parent
REPO=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
S=REPO/'code/bw1000/resume-20261026/Kimi-TP32-I96-template-component-r163'
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt_file=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/next-parallel-read-root-r9487-a.json')
receipt=json.loads(receipt_file.read_bytes())['CPU_TP32']
b=receipt['data']
assert receipt['SHA256']=='30eafbbf4934bd446c1a77a01a12be822e97a7964c2d8b2c58ef830d9ee3f055'
assert b['actual_step']=='910407.517' and b['completed'] and b['CPU_only'] and not b['GPU'] and not b['Native_loaded']
assert b['source_manifest_SHA256']==sha(S/'source-manifest-r163.json')=='e4a2029fb5cc026779567e8c3966e6dd09579ce201d6f9874892ee7293cedb15'
assert b['native_SHA256']=='2e7c52d34c9ec6ccc6306517e77b76f56265e3309071979d43d425bc28254197'
assert b['library_SHA256']=='94d7519e4472e8c4643792a7efedfd644b97978e5e6b2cbf347da29c8debc6f1'
for name,h in json.loads((S/'source-manifest-r163.json').read_bytes())['files_SHA256'].items(): assert sha(S/name)==h,name
for p in [D/'controller_r163a.py',D/'idle_r163a.py',S/'run_GPU_r163.py']: ast.parse(p.read_bytes())
c=(D/'controller_r163a.py').read_text()
for s in ["PARENT='910380'","NODES=['a14r2n08','a14r2n09','a14r2n10','a14r2n12']","os.environ['SLURM_JOB_ID']=='910407'","'-N4','-n4'","'-t','00:12:00'","normal_layer_owner_closes=32","len(owner['arms'])==48","'Using network IBext_v8'","'Using network Socket'not in text","w['post_KFD']=idle('post')","with intent.open('x')","actual.startswith(PARENT+'.')","b['actual_step']=='910407.517'",receipt['SHA256'],b['native_SHA256'],b['library_SHA256']]:assert s in c,s
for s in ['910384','n17','n18','r161b','range(2)','-N2','r161a','subprocess.run([\'ssh\''] :assert s not in c,s
assert c.count("'--gres=dcu:8'")==1
assert c.count("process=subprocess.Popen(args")==1
assert (D/'GPU_inside_r163a.sh').read_text().strip()=='#!/bin/bash\nset -euo pipefail\nexec bash /refactor/incoming/Kimi-TP32-I96-template-component-r163/GPU_inside_r163.sh'
manifest={'revision':'r163a','scope':'one-shot root-owned CPU keeper controller; TP32PP1 realL1 genericI96 versus TT2Tile4I96/W2pair; unchanged compiled r163', 'GPU_started':False,'CPU_keeper':'910407','parent_job':'910380','nodes':['a14r2n08','a14r2n09','a14r2n10','a14r2n12'],'hard_GPU_minutes':12,'CPU_build_path':receipt['path'],'CPU_build_SHA256':receipt['SHA256'],'CPU_build_actual_step':b['actual_step'],'Native_SHA256':b['native_SHA256'],'library_SHA256':b['library_SHA256'],'component_source_manifest_SHA256':b['source_manifest_SHA256'],'files_SHA256':{n:sha(D/n) for n in ['controller_r163a.py','idle_r163a.py','GPU_inside_r163a.sh']}}
(D/'launch-source-manifest-r163a.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
contract={'passed':True,'AST':True,'all_component_source_pins':True,'actual_receipt_517_locked':True,'one_GPU_launch_max':True,'no_compile_or_component_change':True,'component_inner_timeout_s':660,'slurm_GPU_timeout_s':720,'fresh_KFD_before_after':True,'IBext_noSocket':True,'normal_owners_required':32,'held_out_Model_admission':False,'GPU_executed':False,'SSH_Git_Slurm_executed':False,'manifest_SHA256':sha(D/'launch-source-manifest-r163a.json')}
(D/'CPU-local-contract-r163a.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf8')
print(json.dumps(contract))
