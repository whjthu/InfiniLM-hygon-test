"""Re-audit original804 HOST receipt and frozen source with stdlib only."""
from pathlib import Path
import hashlib,json,sys
D=Path(__file__).resolve().parent
MF='e55fe87d2fa283d0829cb78b36f61831e20b91fcb392c3785012e9520cf24930'
RECEIPT='241c1af8a7ff7c7f689ea7e14cc32971beed70aced62e37413c60648400ddc32'
LIB='9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'
sha=lambda b:hashlib.sha256(b).hexdigest()
def read(path):
 raw=Path(path).read_bytes();o=json.loads(raw);f=o['receipt'];q=f['data']
 assert sha(json.dumps(q,indent=2).encode())==f['SHA256']==RECEIPT
 assert q['completed']and q['producer']=='910407.804'and q['CPU_only']and not q.get('error')
 assert q['HOST_TUs_compiled']==1 and q['GPU_TUs_compiled']==0 and q['GPU']==0 and not q['GPU_initialized']and not q['Native_dlopen']and not q['checkpoint_IO']
 assert q['source_manifest_SHA256']==MF and q['library_SHA256']==LIB
 assert q['parent_Native_SHA256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'
 assert len(q['reused_objects_SHA256'])==28 and len(q['parent_objects_SHA256'])==q['old_object_count']==29
 assert len(q['final_objects_SHA256'])==q['final_object_count']==29 and len(q['compiled'])==1
 c=q['compiled'][0];assert c['source'].endswith('/caller_v_on_demand_r403.cpp')and q['final_objects_SHA256'][c['object']]==c['object_SHA256']
 assert q['all_parent_strong_exports_retained']and q['all_parent_CABI_retained']and q['all_parent_exports_retained']and not q['missing_private_weak_exports']and q['ldd_r_parent_equal']
 assert q['class_header_layout_unchanged']and q['default_OFF']and q['added_sync']==0 and not q['provider_GPU_compiled']and q['whole_Model_linked']
 assert not q['actual_device_Model_qualified']and not q['loaded_service_8GiB_floor_changed']and not q['forward_512MiB_floor_changed']and not q['allocator_peak_qualified']and not q['Model_GO']and not q['GPU_GO']
 h=q['actual_HOST_contract'];assert h['passed']and h['strict_01']and h['actual_policy_header']and h['original_C1_batchflag1_inactive']and h['actual_host_ID_container']and h['async_source_after_local_scope_valid']and h['exception_drain_before_host_release']and h['fixed8192_block_order']and not h['Native_executed']
 assert sha((D/'CPU_SOURCE_MANIFEST.json').read_bytes())==MF;m=json.loads((D/'CPU_SOURCE_MANIFEST.json').read_bytes())
 for n,v in m['files_SHA256'].items():assert sha((D/n).read_bytes())==v,n
 assert c['source_SHA256']==sha((D/'caller_v_on_demand_r403.cpp').read_bytes())
 for n,v in m['files_SHA256'].items():
  if n.endswith('.hpp'):assert c['dependencies_SHA256']['/refactor/incoming/GLM-V-on-demand-Model-integration-r403/'+n]==v,n
 assert len([x for x in q['commands']if '-c'in x])==1
 assert len([x for x in q['commands']if any(v.endswith('/CPU_contract.cpp')for v in x)])==1
 assert any(r.split('|')[:5]==['910407.804','GLM-V-on-demand-Model-integration-r403-CPU','COMPLETED','0:0','00:00:21']for r in o['accounting'])
 return dict(read_passed=True,CPU='910407.804',terminal='COMPLETED0/21s',producer_elapsed_s=q['elapsed_seconds'],original_observation=str(Path(path).resolve()),observation_SHA256=sha(raw),receipt_path=f['path'],receipt_SHA256=RECEIPT,library=q['library'],library_SHA256=LIB,source_manifest_SHA256=MF,source_members=len(m['files_SHA256']),parent_Native_SHA256=q['parent_Native_SHA256'],original_objects=29,final_objects=29,reused_objects=28,necessary_Model_caller_HOST_TUs_compiled=1,tiny_HOST_contract_executables_compiled=1,GPU_TUs_compiled=0,Native_dlopen=False,checkpoint_IO=False,GPU=0,Model=False,all_parent_exports_retained=True,missing_private_weak_exports={},ldd_r_parent_equal=True,actual_HOST_contract=h,compiled=[{**{k:v for k,v in c.items()if k!='dependencies_SHA256'},'dependency_count':len(c['dependencies_SHA256']),'dependency_map_SHA256':sha(json.dumps(c['dependencies_SHA256'],sort_keys=True,separators=(',',':')).encode())}],reused_objects_SHA256=q['reused_objects_SHA256'],original_header_dependency_count=len(q['original_headers_and_project_dependencies_SHA256']),original_header_dependency_map_SHA256=sha(json.dumps(q['original_headers_and_project_dependencies_SHA256'],sort_keys=True,separators=(',',':')).encode()),actual_dynamic_dependency_SHA256=q['actual_dynamic_dependency_SHA256'],class_header_layout_unchanged=True,policy='GLM_V_ON_DEMAND_R403 strict0/1 defaultOFF constructor-frozen; same C1/32768/TP8 fullNative indexer and state body',host_ID_copy_cost='Owned HOST copy included in future Model forward; no added sync; r400 component cost is not a Model estimate',memory='8GiB loaded admission and512MiB forward guard unchanged. Loader staging/allocator peak and loaded-service fit UNKNOWN. Original395 snapshot not rewritten by logicalV saving.',limits='FullNative HOST link and policy/lifetime contract only; no C1PP2 actual loaded Model/task/routing/quality qualification.',GPU_GO=False,Model_GO=False,replay=False)
if __name__=='__main__':print(json.dumps(read(sys.argv[1]),indent=2))
