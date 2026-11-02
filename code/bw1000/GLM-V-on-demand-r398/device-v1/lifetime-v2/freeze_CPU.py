from pathlib import Path
import hashlib,json,ast,difflib
D=Path(__file__).resolve().parent;P=D.parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 changes=[]
 for old,new in [('GPU_run.py','GPU_run_lifetime_v2.py'),('GPU_inside.sh','GPU_inside_lifetime_v2.sh'),('controller.py','controller_lifetime_v2.py'),('resource_probe.py','resource_probe_lifetime_v2.py'),('launch_once.py','launch_once_lifetime_v2.py')]:
  (D/new).write_bytes((P/new).read_bytes())
  changes.append(dict(parent=old,parent_SHA256=sha(P/old),new=new,new_SHA256=sha(D/new),diff=list(difflib.unified_diff((P/old).read_text().splitlines(),(P/new).read_text().splitlines(),lineterm=''))))
 (D/'RUNTIME_DERIVATION.json').write_text(json.dumps(changes,indent=2)+'\n',encoding='utf8')
 for p in D.glob('*.py'):ast.parse(p.read_text());compile(p.read_bytes(),str(p),'exec')
 derivation=json.loads((D/'DERIVATION.json').read_bytes());raw=(P/'attention_component_r398.cpp').read_bytes();assert sha(P/'attention_component_r398.cpp')==derivation['parent_SHA256'];s=raw.decode()
 for e in derivation['edits']:assert s.count(e['old'])==1;s=s.replace(e['old'],e['new'])
 assert s.encode()==(D/'attention_component_r398.cpp').read_bytes()and len(derivation['edits'])==3
 for p in D.glob('*.sh'):assert b'\r'not in p.read_bytes()
 local=dict(passed=True,actual_CPU_compile=False,GPU=0,Native=False,source_delta_exact=True,CXX_edits=3,added_sync=0,math_changed=False,old796_source_unchanged=True,HOST_lifetime_proof='SelectionMap is a Work member; Work destructor body drains before members are destroyed even during unwind. All move/assignment is before H2D borrow. No Runtime-exception-drain guarantee assumed.',runtime_changes='Only new artifact/manifest/entry filenames and actual dependency verification; old files retained unchanged; no math/loop/budget change')
 (D/'local-CPU-result.json').write_text(json.dumps(local,indent=2)+'\n',encoding='utf8')
 files=['attention_component_r398.cpp','DERIVATION.json','CPU_build.py','CPU_inside.sh','GPU_run_lifetime_v2.py','GPU_inside_lifetime_v2.sh','controller_lifetime_v2.py','resource_probe_lifetime_v2.py','launch_once_lifetime_v2.py','RUNTIME_DERIVATION.json','freeze_CPU.py','local-CPU-result.json']
 m=dict(schema=1,files_SHA256={n:sha(D/n)for n in files},CPU_claim=dict(parent=910407,node='a14r2n20',cores=4,mem_MiB=8192,total_s=300,inside_s=270,attempts=1,GPU=0,Native_dlopen=False,checkpoint_payload_IO=False,HOST_TUs_compiled=1,GPU_TUs_compiled=0),parent_CPU='910407.796',parent_CPU_receipt_SHA256='3b4e14f883afade96706b26e19fa7a2322903eec7dea92c7bb1df4a456ced24c',GPU_GO=False,Model_GO=False)
 p=D/'CPU_SOURCE_MANIFEST.json';p.write_text(json.dumps(m,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(SHA256=sha(p),members=len(files),CPP_edits=3,GPU_GO=False)))
if __name__=='__main__':main()
