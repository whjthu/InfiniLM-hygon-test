import pathlib,json,hashlib,base64
D=pathlib.Path(__file__).parent;C=D/'controller';repo=D.parents[2];results=repo/'results/bw1000/2026-11-02';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
observed=json.loads((results/'GLM349-whole-glue-v4-observed.json').read_bytes())
for entry in observed['receipts']:
 q=entry['data'];raw=json.dumps(q,indent=2).encode();assert hashlib.sha256(raw).hexdigest()==entry['SHA256']
 step=q['actual_step'].split('.')[-1];(results/('GLM349-whole-glue-CPU'+step+'-'+('PASS'if q['passed']else'FAIL')+'.json')).write_bytes(raw)
entry=max((x for x in observed['receipts']if x['data']['passed']),key=lambda x:int(x['data']['actual_step'].split('.')[-1]));assert entry['data']['completed']
(C/'check_controller_CPU.py').write_bytes((D/'check_controller_CPU.py').read_bytes())
(C/'candidate_binding.py').unlink(missing_ok=True)
for p in C.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
for p in [*C.glob('*.sh'),*D.glob('*.sh')]:assert b'\r'not in p.read_bytes(),str(p)
mf={'files_SHA256':{p.name:sha(p)for p in sorted(C.iterdir())if p.is_file()and p.name!='SOURCE_MANIFEST.json'},'parent_controller':'GLM-prefill-W2-W4-controller-r308','owner_manifest_SHA256':sha(D/'owner/SOURCE_MANIFEST.json'),'whole_CPU_receipt_SHA256':entry['SHA256'],'CPU_scope':'Actual owner/supervisor/client and32owner cleanup statements, recording CAPI/WORLD/HTTP/process only; no GPU or Native Context'}
(C/'SOURCE_MANIFEST.json').write_bytes((json.dumps(mf,indent=2)+'\n').encode())
go=json.loads((C/'execution-GO.template.json').read_bytes());go.update(GPU_authorized=True,publication_commit='a'*40,controller_manifest_SHA256=sha(C/'SOURCE_MANIFEST.json'),whole_CPU_receipt_path=entry['path'],whole_CPU_receipt_SHA256=entry['SHA256'])
(D/'private-execution-GO.prepublish.json').write_bytes((json.dumps(go,indent=2)+'\n').encode())
files={p.name:base64.b64encode(p.read_bytes()).decode()for p in C.iterdir()if p.is_file()}
source="""import pathlib,json,base64,hashlib,sys,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';d=r/'incoming/GLM-W4-cross-workload-r349/controller_v3';assert not d.exists();d.mkdir()
files=FILES
for n,b in files.items():(d/n).write_bytes(base64.b64decode(b))
sys.path.insert(0,str(d));import controller_r349 as c
go=GO
c.verify_sources(go);c.require_baseline_terminal(r,go)
z=subprocess.run([sys.executable,'-B',str(d/'check_controller_CPU.py')],capture_output=True,text=True,timeout=30)
q=json.loads(z.stdout.strip().splitlines()[-1]);assert z.returncode==0 and q['completed']and q['passed']
print(json.dumps({'source_pins_verified':True,'GPU_submitted':False,'controller_manifest_SHA256':hashlib.sha256((d/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),'owner_manifest_SHA256':c.OWNER_SHA,'whole_CPU_receipt_SHA256':go['whole_CPU_receipt_SHA256'],'recording_controller_CPU':q}))
""".replace('files=FILES','files='+repr(files)).replace('go=GO','go='+repr(go))
compile(source,'remote_upload_controller_v3.py','exec')
(D/'remote_upload_controller_v3.py').write_bytes(source.encode())
claim={'state':'SOURCE_CPU_CLOSED_GPU_NOT_SUBMITTED','owner':'/root/glm_operator_tuning','runtime':'GLM-W4-cross-workload-r349/runtime_v4/owner','source_manifest_SHA256':sha(D/'owner/SOURCE_MANIFEST.json'),'controller':'GLM-W4-cross-workload-r349/controller_v3','controller_manifest_SHA256':sha(C/'SOURCE_MANIFEST.json'),'whole_CPU_actual':entry['data']['actual_step'],'whole_CPU_receipt_SHA256':entry['SHA256'],'CPU673_official_encoding_PASS':True,'GPU_parent':910376,'nodes':['a02r4n12','a02r4n13','a02r4n14','a02r4n15'],'cards_per_node':8,'TP':8,'PP':4,'context':32768,'HTTP_and_Native_requests':18,'fixtures':[{'id':f['id'],'tokens':f['current_prepared_prompt_tokens'],'request_SHA256':f['request_SHA256'],'prompt_IDs_SHA256':f['current_prepared_prompt_IDs_SHA256'],'windows':[2,4,2,4,4,2]}for f in json.loads((D/'owner/FIXTURE_CARD.json').read_bytes())['fixtures']],'ordinary_output_budget':4096,'GPU_max_s_including_load':10800,'Native_original_start_max_s':10200,'service_original_start_s':9120,'CPU_keeper_parent':910407,'CPU_keeper_node':'a14r2n20','CPU_keeper_max_s':12000,'CPU_keeper_cores':1,'CPU_keeper_mem_MiB':4096,'GPU_stage_cores':64,'GPU_stage_mem_MiB':128000,'Native_DSO_SHA256':'038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00','one_residency':True,'no_retry_reload':True,'no_new_CXX_build':True,'no_decode_math_SDK_change':True,'fresh_exact_parent_idle_KFD_dedup_before_start':True,'all32_normal_destroy_and_post_KFD_required':True,'raw_payloads_and_prompt_ID_sequences_stay_private':True,'anchor_is_cross_residency_control_not_independent_new_gain':True}
(results/'GLM349-final-source-CPU-resource-claim.json').write_bytes((json.dumps(claim,indent=2)+'\n').encode())
print(json.dumps({'controller_manifest_SHA256':sha(C/'SOURCE_MANIFEST.json'),'owner_manifest_SHA256':claim['source_manifest_SHA256'],'CPU677_SHA256':entry['SHA256'],'GPU_GO':False}))
