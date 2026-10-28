"""Local freezer, no dispatch. Preserve root's separate final execution decision."""
from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent;M=D/'model';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();prefix='/refactor/incoming/Kimi-sparse-profile-Model-r162/'
p=M/'controller-contract-r136.json';c=json.loads(p.read_bytes());c.update(native='798c4bca01cddb0755a91171ef61dfa755329518bb8cfc87640305a24e55a6e1',native_recompiled='CPU511 host worker/weights only; no new GPU TU',profile='loading process CPU/faultIO/registry; no sparse GPU profile requested',pilot='C1 four questions plus C8 two groups; realEOS cap512; GSM16 reused screen',cache='fixed ON before Runtime',W13='fixed OFF, actual branch counters must prove zero candidate calls',GPU_GO_present=False,actual_CPU_preflight_pending=True);p.write_text(json.dumps(c,indent=2)+'\n')
p=M/'CPU-source-contract-r136.json';c={'CPU_source_only':True,'GPU_qualified':False,'actual_CPU_preflight':'PENDING','actual_gpu':'NOT_SUBMITTED','case_count':6,'request_executions':20,'GSM_unique':16,'not_new32_heldout':True,'C1_subset_of_C8':True,'W13':'fixedOFF','loading_cache':'fixedON','allpyAST':True,'shell_syntax':'parent verified shells retained; target_inside adds only two literal exports','parent_sparse_profile_mock_contract_provenance':'retained in original136b; not this pilot profile qualification'};p.write_text(json.dumps(c,indent=2)+'\n')
p=M/'runtime/preflight_r9183.py';text=p.read_text()
text=text.replace('import target_protocol,entry_contract_r9143,parameter_loader_r9143,native_binding_r9143,host_observer_r9183,shard_head_contract_r9183','import target_protocol,entry_contract_r9143,parameter_loader_r9143,native_binding_r9143,host_observer_r9183,shard_head_contract_r9183,model_registry_contract_r162,loader_work_observer_r158')
text=text.replace('native_binding_r9143,host_observer_r9183,shard_head_contract_r9183]:','native_binding_r9143,host_observer_r9183,shard_head_contract_r9183,model_registry_contract_r162,loader_work_observer_r158]:')
text=text.replace("for name in ['modern_kimi_model_events_configure_r8669zq'","for name in ['modern_kimi_loading_registry_work_r159','modern_kimi_loading_binding_counts_r9212','modern_kimi_W13_wavepartial_counts_r155','modern_kimi_model_events_configure_r8669zq'")
p.write_bytes(text.encode())
mp=M/'runtime/model-manifest-r9183.json';m=json.loads(mp.read_bytes())
for path in m['source_SHA256']:
 if path.startswith(prefix):m['source_SHA256'][path]=sha(M/path[len(prefix):])
mp.write_text(json.dumps(m,indent=2)+'\n')
go=M/'launcher/GPU-GO-schema.json';g=json.loads(go.read_bytes());g['manifest_SHA256']=sha(mp);go.write_text(json.dumps(g,indent=2)+'\n')
pin=M/'launcher/model-source-pins.json';pins=json.loads(pin.read_bytes());pins['manifest_SHA256']=sha(mp);pins['files']={prefix+str(f.relative_to(M)).replace('\\','/'):sha(f)for f in M.rglob('*')if f.is_file()and f!=pin and f.name!='producer-source-r162.json'};pin.write_text(json.dumps(pins,indent=2)+'\n')
p=M/'producer-source-r162.json';producer=json.loads(p.read_bytes());producer['manifest_SHA256']=sha(mp);producer['files_SHA256']={str(f.relative_to(M)).replace('\\','/'):sha(f)for f in M.rglob('*')if f.is_file()and f!=p};p.write_text(json.dumps(producer,indent=2)+'\n')
for p in M.rglob('*.py'):ast.parse(p.read_text())
print(json.dumps({'manifest_SHA256':sha(mp),'producer_SHA256':sha(M/'producer-source-r162.json'),'source_pins_SHA256':sha(pin)}))
