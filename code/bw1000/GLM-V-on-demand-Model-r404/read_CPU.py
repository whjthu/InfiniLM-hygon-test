"""Stdlib audit of actual813; CPU bridge proof, never a Model math verdict."""
from pathlib import Path
import hashlib,json,sys
D=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
RECEIPT='6f9972f0651b51b4923b0180f3abf3eb35b1313ac9c1a457be29eeaa9de22ffe'
PARENT='a3de385519b8e5ce238d08dcd2ae2ad153c8ac5d42fbeb925164bbb36c408477'
V3='8fa913f13fd177953b75c8bbda96372df778f25aca72797ace8338e50988e70c'
OWNER='c57fcf4929391d0bde21e1cdf176c6da65787cd48230c2973c36814f4d5fe3fa'
def read(path):
 raw=Path(path).read_bytes();o=json.loads(raw);f=o['receipt'];q=f['data']
 assert sha((json.dumps(q,indent=2)+'\n').encode())==f['SHA256']==RECEIPT
 assert q['completed']and q['passed']and q['actual_CPU']=='910407.813'and q['GPU']==q['Model']==0 and not q['Native_dlopen']and not q['weights']and not q['compile']
 assert q['source_manifest_SHA256']==PARENT and q['CPU_source_manifest_SHA256']==V3 and q['owner_manifest_SHA256']==OWNER
 for p,h in [(D/'SOURCE_MANIFEST.json',PARENT),(D/'owner/SOURCE_MANIFEST.json',OWNER),(D/'cpu-v3/CPU_SOURCE_MANIFEST.json',V3)]:
  assert sha(p.read_bytes())==h
  for n,digest in json.loads(p.read_bytes())['files_SHA256'].items():assert sha((p.parent/n).read_bytes())==digest,n
 body=q['whole_entry_result'];assert body['completed']and body['passed']and body['GPU']==body['Model']==body['Native_Context']==0 and not body['actual_HTTP']and not body['real_TCP_listener']
 assert body['root_core_llm_guarded']and not body['actual_root_or_ABI_qualified']and not body['real_GPU_GO']
 n=body['actual_r403_source_graph'];assert n['library_SHA256']=='9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'and n['cpu_receipt_SHA256']=='241c1af8a7ff7c7f689ea7e14cc32971beed70aced62e37413c60648400ddc32'
 entry=body['actual_samectx_changed_entry'];assert entry['passed']and entry['same_ctx']and entry['actual_OwnedContext']and entry['actual_constructor_snapshot']and entry['actual_pinned_loader_body']and entry['original_admission_AST']and entry['original_worker_create_calls']==1
 trace=q['phase_trace'];assert trace['completed']and len(trace['cases'])==8 and all(c['passed']for c in trace['cases'])
 neg=trace['original_leaf_regression'];assert neg['rejected']and not neg['GPU_initialized']and'GPU forbidden'in neg['error']
 assert sha((json.dumps(trace,indent=2)+'\n').encode())==q['phase_trace_SHA256']==entry['phase_trace_SHA256']
 expected=['normal','census_reject','floor_reject','load_failure','finish_failure','request_failure','cleanup_failure','primary_plus_cleanup_failure'];assert[c['name']for c in trace['cases']]==expected
 for c in trace['cases']:
  assert c['counts']['create']==c['counts']['destroy']==c['counts']['shutdown']==1
  l=c['lifecycle'];assert l['completed']and l['destroy_attempts']==l['shutdown_attempts']==1
  if c['name']in['census_reject','floor_reject']:assert c['counts']['load']==c['counts']['finish']==0
  if c['name']=='normal':assert l['normal']and c['counts']['load']==c['counts']['finish']==1
  if c['name']=='primary_plus_cleanup_failure':assert l['primary_error']and l['cleanup_errors']
 cases=body['cases'];em=next(c for c in cases if c['name']=='actual_emitter_owner_to_actual_supervisor_and_client_main');assert em['requests']==em['fixtures']==2 and em['worker_identity_complete']
 owners=next(c for c in cases if c['name']=='all16_actual_owner_loops_and_actual_worker_cleanup_suffix')['owners'];assert len(owners)==16 and{(v['stage'],v['rank'])for v in owners}=={(s,r)for s in range(2)for r in range(8)}and all(v['completed']and v['requests']==2 and v['destroy']==v['shutdown']==1 for v in owners)
 control=q['actual_controller_main_recording'];assert control['passed']and len(control['cases'])==2 and all(c['Popen_count']==1 and c['retries']==0 and c['actual_handle_preserved']for c in control['cases'])
 assert any(r.split('|')[:5]==['910407.813','GLM-r404-whole-entry-CPU-v3','COMPLETED','0:0','00:00:24']for r in o['accounting'])
 return dict(read_passed=True,actual_CPU='910407.813',terminal='COMPLETED0/24s',elapsed_s=q['elapsed_s'],receipt_path=f['path'],receipt_SHA256=RECEIPT,original_observation=str(Path(path).resolve()),observation_SHA256=sha(raw),parent_source_manifest_SHA256=PARENT,CPU_v3_source_manifest_SHA256=V3,owner_manifest_SHA256=OWNER,Native_SHA256=n['library_SHA256'],Native_CPU804_receipt_SHA256=n['cpu_receipt_SHA256'],phase_trace_path=q['phase_trace_path'],phase_trace_SHA256=q['phase_trace_SHA256'],phase_summaries=[dict(name=c['name'],passed=c['passed'],counts=c['counts'],primary=c['lifecycle']['primary_error'],cleanup_errors=c['lifecycle']['cleanup_errors'])for c in trace['cases']],original_guarded_leaf_reproduction=neg,original809_primary_reconstructed=False,actual_prepared_owner_client_bridge=dict(owners=16,logical_requests=2,fixtures=[8900,9741],prefill_window=2,Native_CAPI_recording=True,HTTP_transport_recording=True,real_prepared_queue=True,actual_worker_cleanup_suffix=True),actual_controller_recording=control,Native_dlopen=False,weights=False,compile=False,GPU=0,Model=0,production_owner_changed=False,original_CPU804_repeated=False,original_failures=['809FAILED','812FAILED'],Model_math_or_physical_fit_qualified=False,GPU_GO=False,Model_GO=False,scope='Changed-glue CPU source/identity/lifetime proof only. Native/weights/HTTP/Slurm execution leaves are recording; no device/task/fit or speed claim.')
if __name__=='__main__':print(json.dumps(read(sys.argv[1]),indent=2))
