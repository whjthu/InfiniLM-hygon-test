"""Read the original bounded two-TU CPU artifacts; no compiler/Native/SDK calls.
Current project objects/DSOs are independently hashed. Non-project dependency
bytes retain producer snapshots, distinct from a full current toolchain replay.
"""
from pathlib import Path
import argparse,hashlib,json,shlex,re
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
D=R/'incoming/Kimi-vocab-parallel-embedding-r461/split-v128-pack-reuse-CPU-v1'
B=R/'candidates/Kimi-split-V128-pack-reuse-CPU-v1'
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def project(p):
 if p.startswith('/refactor/'):return R/p[len('/refactor/'):]
 p=Path(p);assert p.is_absolute()and p.is_relative_to(R);return p
def deps(p):return set(shlex.split(Path(p).read_text().replace('\\\n',' ').partition(':')[2]))
def strong(t):return {a.split()[-1]:a.split()[-2]for a in t.splitlines()if len(a.split())>=3 and a.split()[-2]in ['T','D','B','R']}
def needed(t):return re.findall(r'\(NEEDED\).*Shared library: \[([^]]+)\]',t)
def main(receipt_sha):
 rp=B/'CPU-result.json';assert re.fullmatch('[0-9a-f]{64}',receipt_sha)and H(rp)==receipt_sha;q=json.loads(rp.read_bytes());assert q['completed']and q['passed']and q['new_HOST']==1 and q['new_GPU']==1 and q['all_fresh_M_before_compile']and q['all_fresh_M_equal_MD']and not q['unmatched_dependencies']and not q['project_maps']
 assert H(D/'SOURCE_MANIFEST.json')==q['source_manifest_SHA256']=='b3ffbfb9abb7d0620934458cfac5a3e4d967114f35e751f701844fb5407e1e2f';assert H(D/'CPU_RESOURCE_CLAIM.json')==q['claim_SHA256']=='12a94146900edab3ad2ceacacc06f4743f1a198a9ea45e98e5effe847617e010'
 mf=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(H(D/p)==h for p,h in mf['files_SHA256'].items());plan=json.loads((D/'BUILD_PLAN.json').read_bytes())
 def log(label,stream='stdout'):
  v=q['logs'][label+'.'+stream];p=project(v['path']);assert H(p)==v['SHA256']and p.stat().st_size==v['bytes'];return p.read_text()
 changed=[];objects={}
 for u in plan['units']:
  n=u['name'];v=q['compiled'][n];m=project(v['MD']);before=B/(n+'.before.d');assert deps(m)==deps(before)==set(q['fresh_dependencies_SHA256'][n])and H(m)==v['MD_SHA256'];assert H(project(v['source']))==v['source_SHA256']==H(D/u['source']);assert H(project(v['object']))==v['object_SHA256'];objects[v['object']]=v['object_SHA256'];assert str(Path('/refactor/incoming/Kimi-vocab-parallel-embedding-r461/split-v128-pack-reuse-CPU-v1/native-src/fa_kernels.h'))in q['fresh_dependencies_SHA256'][n]
  changed.append(dict(name=n,kind=u['kind'],source_SHA256=v['source_SHA256'],object_SHA256=v['object_SHA256'],MD_SHA256=v['MD_SHA256'],original_M_first_observed_SHA256=H(before),dependency_count=len(deps(m))))
 for p,h in plan['objects_SHA256'].items():assert H(project(p))==h;objects[p]=h
 assert len(objects)==37
 libs={};checked_project={}
 for item in plan['links']:
  n=item['name'];v=q['libraries'][n];p=project(v['path']);assert H(p)==v['SHA256'];actual=strong(log(n+'-symbols'));old=strong(log(n+'-parent-symbols'));assert all(actual.get(k)==t for k,t in old.items())and len(old)==v['strong_parent_exports_retained'];assert all(actual.get(k)=='T'for k in item.get('required_exports',[]))
  original_needed=needed(log(n+'-parent-ELF'));expected=[q['libraries'][item.get('replace_resolution',{}).get(k,'')]['path']if k in item.get('replace_resolution',{})else k for k in original_needed];assert needed(log(n+'-ELF'))==v['NEEDED']==expected
  output=log(n+'-ldd')+'\n'+log(n+'-ldd','stderr');assert 'not found'not in output and 'undefined symbol'not in output;paths=set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)',output,re.M));assert paths==set(v['all_transitive_resolved_library_SHA256'])
  for path,h in v['all_transitive_resolved_library_SHA256'].items():
   if path.startswith('/refactor/')or path.startswith(str(R)+'/'):assert H(project(path))==h;checked_project[path]=h
  assert all(objects[k]==h for k,h in v['all_linked_object_SHA256'].items())and len(v['all_linked_object_SHA256'])==item['object_count'];cmd=next(c['argv']for c in q['commands']if c['name']==n+'-link');o=[i for i,a in enumerate(cmd)if a.endswith('.o')];dsos=[i for i,a in enumerate(cmd)if a.startswith('-l')or a in [x['path']for x in q['libraries'].values()]];assert max(o)<min(dsos)
  libs[n]=dict(path=v['path'],SHA256=v['SHA256'],bytes=p.stat().st_size,object_count=item['object_count'],parent_strong_exports_retained=len(old),NEEDED=v['NEEDED'],resolved_transitive=len(paths))
 return dict(schema=1,actual_CPU=q['actual_CPU'],passed=True,receipt_SHA256=receipt_sha,source_manifest_SHA256=q['source_manifest_SHA256'],claim_SHA256=q['claim_SHA256'],elapsed_inside_s=q['elapsed_s'],changed_TUs=changed,reused_production_objects=34,reused_probe_objects=1,live_objects_verified=len(objects),libraries=libs,live_project_DSO_SHA256=checked_project,Native_dlopen=False,SDK_calls=0,GPU_GO=False,Model_GO=False,numeric_qualified=False,limits=['Current non-project headers/compiler/DTKDSO bytes were not independently rehashed; producer snapshots retained.','Original before-M SHA first computed here; producer saved MD and full freshdependency sets.','C++/ELF/loader qualification only; actual shape1 offsetviews/vendor workspace/causal numeric/driver peak/performance remain UNKNOWN.'])
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--receipt-sha',required=True);x=a.parse_args();print(json.dumps(main(x.receipt_sha)))
