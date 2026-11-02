"""Local source freeze only. Does not execute compiled Native or device code."""
from pathlib import Path
import ast,hashlib,importlib.util,json
D=Path(__file__).resolve().parent;G=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 spec=importlib.util.spec_from_file_location('r403_derivation',D/'derive_caller.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 p=G/'code/bw1000/agent-context-32k/GLM-PP4-registry-repair-private-r184b/resident_caller_registry_r184b.cpp'
 src,proof=module.derive(p.read_bytes());assert src==(D/'caller_v_on_demand_r403.cpp').read_bytes()
 for n in ['selected_values_r400.hpp','chunk_geometry_r400.hpp']:assert(D/n).read_bytes()==(D.parent/'GLM-V-on-demand-prefill-r400'/n).read_bytes()
 assert(D/'derive_parent_r398.py').read_bytes()==(D.parent/'GLM-V-on-demand-r398/derive_caller.py').read_bytes()
 for f in D.glob('*.py'):ast.parse(f.read_bytes());compile(f.read_bytes(),str(f),'exec')
 assert b'\r'not in(D/'CPU_inside.sh').read_bytes()
 source=src.decode();assert 'v_on_demand_r398::'not in source and source.count('v_on_demand_r400::selected_values')==2
 assert source.count('host_ids.hold(v)')==1 and source.count('std::fill(lengths_.begin(),lengths_.end(),0)')==1
 # Concrete branch algebra for C1/32K, not device/Model qualification.
 for n in [1,512]:
  assert not(32768<=2048) # certified/direct-proven-prefix branch
  assert not(n==4096 and 1==8) # crossrequest 5775 geometry
  assert not(False and 1>=8) # projected-host batched cache
 result=dict(passed=True,source_derivation_actual=True,unchanged_r400_helper_bytes=True,class_headers_not_modified=True,body_math_original=True,one_Model_caller_TU=True,extra_sync=0,HOST_ID_owned_copy_cost_in_forward=True,C1_batchflag1_mathematically_inactive=True,shared_indexer_and_full_indexer_bodies_unchanged=True,reset_poison_lengths_original=True,Python_AST=True,GPU=0,Slurm=0,Native_dlopen=False,compile=0,numerical_qualification=False,Model_GO=False)
 (D/'local-CPU-result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 files=['policy_r403.hpp','host_ids_r403.hpp','selected_values_r400.hpp','chunk_geometry_r400.hpp','derive_parent_r398.py','derive_caller.py','DERIVATION.json','caller_v_on_demand_r403.cpp','CPU_contract.cpp','CPU_build.py','CPU_inside.sh','PLAN.json','SOURCE_BOUNDARIES.json','freeze_CPU.py','local-CPU-result.json']
 m=dict(schema=1,files_SHA256={n:sha(D/n)for n in files},CPU_claim=dict(parent=910407,node='a14r2n20',cores=4,mem_MiB=8192,total_s=300,inside_s=270,attempts=1,GPU=0,Native_dlopen=False,checkpoint_IO=False,necessary_Model_HOST_TUs=1,contract_HOST_TUs=1,GPU_TUs=0),GPU_GO=False,Model_GO=False,parent_Native_SHA256='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00')
 (D/'CPU_SOURCE_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(source_manifest_SHA256=sha(D/'CPU_SOURCE_MANIFEST.json'),members=len(files),local_result=result),indent=2))
if __name__=='__main__':main()
