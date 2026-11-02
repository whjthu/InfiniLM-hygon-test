"""Freeze private changed-entry source; does not activate a Model/GPU GO."""
from pathlib import Path
import ast,hashlib,json,importlib.util
D=Path(__file__).resolve().parent;O=D/'owner';G=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert(D/'PLAN.json').is_file()
 base=G/'code/bw1000/GLM-W4-cross-workload-r349/owner';old=json.loads((base/'SOURCE_MANIFEST.json').read_bytes())
 assert sha(base/'SOURCE_MANIFEST.json')=='4e5bbea816c8569fd7fbf382f2efc44787f8937dcf096fbacb889b1c8b917366'
 for n,h in old['files_SHA256'].items():assert sha(base/n)==h,n
 changed={n:dict(old=h,new=sha(O/n))for n,h in old['files_SHA256'].items()if sha(O/n)!=h}
 unchanged={n:h for n,h in old['files_SHA256'].items()if sha(O/n)==h}
 proof=json.loads((D/'OWNER_DERIVATION.json').read_bytes());proof.update(actual_changed_parent_members=changed,unchanged_parent_files=unchanged,actual_added_members=sorted(str(p.relative_to(O)).replace('\\','/')for p in O.rglob('*')if p.is_file()and str(p.relative_to(O)).replace('\\','/')not in old['files_SHA256']and p.name!='SOURCE_MANIFEST.json'))
 (D/'OWNER_DERIVATION.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
 for p in D.rglob('*.py'):ast.parse(p.read_bytes());compile(p.read_bytes(),str(p),'exec')
 for p in D.rglob('*.sh'):assert b'\r'not in p.read_bytes(),str(p)
 ownerfiles={str(p.relative_to(O)).replace('\\','/'):sha(p)for p in O.rglob('*')if p.is_file()and p.name!='SOURCE_MANIFEST.json'}
 om=dict(files_SHA256=ownerfiles,parent_manifest_SHA256=sha(base/'SOURCE_MANIFEST.json'),default_Native_policy_OFF=True,explicit_entry_policy='GLM_V_ON_DEMAND_R403=1',GPU_GO=False,Model_GO=False)
 (O/'SOURCE_MANIFEST.json').write_text(json.dumps(om,indent=2)+'\n',encoding='utf-8')
 spec=importlib.util.spec_from_file_location('r404_controller_check',D/'check_controller_CPU.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
 import sys
 sys.path.insert(0,str(D));recorded=c.run();assert recorded['passed']
 (D/'local-controller-CPU-result.json').write_text(json.dumps(recorded,indent=2)+'\n',encoding='utf-8')
 worker=next(n for n in ast.parse((O/'model_entry_r290.py').read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='worker')
 assert len([n for n in ast.walk(worker)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='modern_glm_model_create'])==1
 lease=next(n for n in worker.body if isinstance(n,ast.With)and isinstance(n.items[0].context_expr,ast.Name)and n.items[0].context_expr.id=='_owned_r404')
 assert any(isinstance(n,ast.Name)and n.id=='constructor_snapshot'for n in ast.walk(lease))
 assert "8 * 1024 ** 3"in ast.unparse(lease)and "512*1024**2"in(O/'owner_r290.py').read_text()
 card=json.loads((O/'FIXTURE_CARD.json').read_bytes());assert[f['current_prepared_prompt_tokens']for f in card['fixtures']]==[8900,9741]
 assert all(f['original_budget']==4096 and f['current_prepared_prompt_tokens']+4096<=32768 for f in card['fixtures'])
 sourcefiles={str(p.relative_to(D)).replace('\\','/'):sha(p)for p in D.rglob('*')if p.is_file()and p.name not in ['SOURCE_MANIFEST.json','GPU_MANIFEST.json']}
 sourcefiles['owner/SOURCE_MANIFEST.json']=sha(O/'SOURCE_MANIFEST.json')
 manifest=dict(schema=1,files_SHA256=sourcefiles,owner_manifest_SHA256=sha(O/'SOURCE_MANIFEST.json'),CPU_claim=dict(parent=910407,node='a14r2n20',cores=4,mem_MiB=8192,total_s=300,inside_s=270,attempts=1,GPU=0,Native_dlopen=False,weights=False,compile=False),GPU_GO=False,Model_GO=False)
 (D/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'),owner_manifest_SHA256=sha(O/'SOURCE_MANIFEST.json'),root_members=len(sourcefiles),owner_members=len(ownerfiles),local_Popen_CPU=recorded['passed'],Slurm=0,GPU=0,Model=0),indent=2))
if __name__=='__main__':main()
