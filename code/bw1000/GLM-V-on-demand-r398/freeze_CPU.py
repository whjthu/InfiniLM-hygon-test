from pathlib import Path
import ast,hashlib,json,sys
from consumer import recording_contract
from derive_caller import main as derive
D=Path(__file__).resolve().parent;G=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 derivation=derive()
 for p in D.glob('*.py'):ast.parse(p.read_text());compile(p.read_bytes(),str(p),'exec')
 contract=recording_contract()
 result=dict(completed=True,passed=True,actual_consumer_recording=contract,Python_syntax=True,actual_CPP_compile=False,GPU=0,Native_dlopen=False,weights=False,production_GPU_execution=False,Model_GO=False,caller_derivation_SHA256=derivation['derived_SHA256'],scope='Local actual schedule with explicitly recording leaves, not Native/math/device proof')
 (D/'local-CPU-result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
 names=['selected_values_r398.hpp','derive_caller.py','caller_v_on_demand_r398.cpp','DERIVATION.json','attention_component_r398.cpp','consumer.py','CPU_selection_contract.cpp','CPU_build.py','CPU_inside.sh','PLAN.json','freeze_CPU.py','local-CPU-result.json']
 q=dict(schema=1,files_SHA256={n:sha(D/n)for n in names},CPU_claim=dict(parent=910407,node='a14r2n20',cores=4,host_mem_MiB=8192,total_s=300,inside_s=270,attempts=1,GPU=0,Native_dlopen=False,checkpoint_IO=False,HOST_TUs_compiled=2,GPU_TUs_compiled=0),GPU_GO=False,Model_GO=False,parent_native_SHA256='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00',parent_receipt_SHA256='6aa62f788aad6080027d665f30d7c9824c07cb1866de905397fef15c56b26529',current_caller_SHA256='8aa1de7218120cc8d6687fbc3110c419d8651028f6cac92982b03fd5e7578dcd',current_prepare_SHA256='2fa890bcb80c4accd41a7d09a658d557535c4922a6d6dea3b672dcec904f972c',class_header_layout_unchanged=True,loaded_service_8GiB_floor_changed=False)
 p=D/'CPU_SOURCE_MANIFEST.json';p.write_text(json.dumps(q,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(manifest_SHA256=sha(p),members=len(names),local_consumer_cases=contract['actual_consumer_cases'])))
if __name__=='__main__':main()
