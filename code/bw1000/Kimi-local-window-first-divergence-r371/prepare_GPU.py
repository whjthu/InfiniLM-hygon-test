from pathlib import Path
import json,hashlib,base64,ast,subprocess,sys
D=Path(__file__).resolve().parent;G=D.parents[2];L=G.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cpu=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==h for n,h in cpu['files_SHA256'].items())
data=json.loads((L/'Kimi371-CPU-first-observed.json').read_bytes());data.pop('log_tail',None);q=data['receipt']
assert q['completed']and q['passed']and q['actual_CPU']=='910407.740'and q['checks']['protocol_commands']==69
assert q['source_manifest_SHA256']==sha(D/'SOURCE_MANIFEST.json')
O=G/'results/bw1000/2026-11-02';(O/'Kimi371-CPU740-firstlayer-and-width-PASS.json').write_text(json.dumps(data,indent=2)+'\n')
p=subprocess.run([sys.executable,'-B',str(D/'check_launch_CPU.py')],capture_output=True,text=True,check=True);local=json.loads(p.stdout);assert local['passed']and local['GPU']==0
(O/'Kimi371-launch-once-local-CPU-PASS.json').write_text(json.dumps(local,indent=2)+'\n')
names=list(cpu['files_SHA256'])+['SOURCE_MANIFEST.json','GPU_driver.py','controller.py','idle.py','GPU_inside.sh','launch_once.py','check_launch_CPU.py']
for n in names:
 path=D/n
 if path.suffix=='.py':ast.parse(path.read_text())
 if path.suffix=='.sh':assert b'\r'not in path.read_bytes()
old=json.loads((D.parent/'Kimi-local-context-crop-r369/gpu-v1/GPU_MANIFEST.json').read_bytes())
cp='/refactor/candidates/Kimi-local-window-first-divergence-r371/CPU-result.json';deps=dict(cpu['dependencies_SHA256']);deps[cp]=data['receipt_SHA256']
claim=dict(owner='/root/kimi_operator_tuning',GPU_parent=910371,GPU_node='a02r3n15',cards=1,CPUs=8,memory_MiB=65536,GPU_max_s=600,original_driver_deadline_s=570,CPU_keeper_parent=910407,CPU_keeper_node='a14r2n20',CPU_keeper_CPUs=1,CPU_keeper_memory_MiB=4096,CPU_keeper_max_s=900,target_cards=0,one_attempt=True,profile=False,L=32638,raw_width=35840,periodic_seed324=True,append_prompt=64,full_OFF_queries=1,full_ON_queries=0,firstattention_full_replay=1,fixed_QKV_crop_SDPA=1,firstattention_crop_recompute=1,protocol_commands=69,begin=1,seal=1,retire=1,close=1,GuardedCodec_fences_expected=72,feature_bytes=32638*35840*2,original_threshold=.02,no_ABBA=True,no_full5layer_ON=True,no_target_reload=True,no_window_grid=True,no_retry=True,numerical_stop='Save scalar source/finite/NRMSE/maxabs/RMS before stop. Invalid OFFreplay>.02/nonfinite errors fail. First observed nonzero retainedproducer stops summaries/further work after the prescribed twoisolatedbranches; no additionalqueries.',memory='Full materializedfirstlayerQKV ~1.34GB + retainedproducer andcrop controls live onlyinside first_layer; original4GiBGuardedCodec floors on all controls; no raw activationdisk/Git',qualification='Causaldiagnostic only. FullOFFquery wall includes allpack/layers/head/selector/finite/fence plus2smallcaptureclones. Isolatedcontrol walls contain capture/copy/fences; no speed/Model/acceptance claim',cleanup='Known originalGPU/keeper/srun terminal, protocolretireclose on diagnosticstop, Runtimeclose, original and independentCPU KFD/PID/ports46947,46962')
m=dict(GPU_GO=True,files_SHA256={n:sha(D/n)for n in names},dependencies_SHA256=deps,CPU_receipt=cp,CPU_receipt_SHA256=data['receipt_SHA256'],actual_CPU='910407.740',CPU_source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'),worker_manifest_SHA256=old['worker_manifest_SHA256'],outer_wrapper_SHA256=old['outer_wrapper_SHA256'],claim=claim,scope='Immutable369 firstlayer function/actualstrict362bRuntime+333 session, new fixed sameinput causal captures; CPU740 actualtiny69cmd+productionwidth sample; realtrainedGPUunknown')
(D/'GPU_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');h=sha(D/'GPU_MANIFEST.json')
tpath=G/'coordination/tasks/Kimi-local-window-first-divergence-r371.json';t=json.loads(tpath.read_bytes());t.update(state='CPU740_PASS_FIXED_FIRSTLAYER_ONE_DRAFT_CAUSAL_CLAIM',GPU_GO=True,new_GPU_submitted=False,CPU740=dict(actual='910407.740',receipt_SHA256=data['receipt_SHA256'],result='results/bw1000/2026-11-02/Kimi371-CPU740-firstlayer-and-width-PASS.json'),GPU_manifest_SHA256=h,once_GPU_claim=claim);tpath.write_text(json.dumps(t,indent=2)+'\n')
new=[n for n in names if n not in cpu['files_SHA256']and n!='SOURCE_MANIFEST.json']+['GPU_MANIFEST.json']
payload={n:base64.b64encode((D/n).read_bytes()).decode()for n in new}
script="""from pathlib import Path
import json,hashlib,base64
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-local-window-first-divergence-r371'
assert hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()==__CPUHASH__
c=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==h for n,h in c['files_SHA256'].items())
for n,b in __PAYLOAD__.items():
 p=D/n;assert not p.exists();p.write_bytes(base64.b64decode(b))
assert hashlib.sha256((D/'GPU_MANIFEST.json').read_bytes()).hexdigest()==__HASH__
m=json.loads((D/'GPU_MANIFEST.json').read_bytes());assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==h for n,h in m['files_SHA256'].items())
print(json.dumps(dict(uploaded=True,GPU_submitted=False,GPU_manifest_SHA256=__HASH__,CPU740_source_unchanged=True)))
""".replace('__CPUHASH__',repr(sha(D/'SOURCE_MANIFEST.json'))).replace('__HASH__',repr(h)).replace('__PAYLOAD__',repr(payload))
ast.parse(script);(L/'remote371-upload-final.py').write_bytes(script.encode())
print(json.dumps(dict(GPU_manifest_SHA256=h,CPU740_receipt_SHA256=data['receipt_SHA256'],claim=claim,helper_AST_passed=True)))
