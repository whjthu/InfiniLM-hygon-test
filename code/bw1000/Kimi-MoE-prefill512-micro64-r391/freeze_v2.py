from pathlib import Path
import hashlib,json,subprocess,sys,datetime
D=Path(__file__).resolve().parent;N=D/'v2';G=D.parents[2];L=G.parent;E=G/'results/bw1000/2026-11-02';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(D/'SOURCE_MANIFEST.json')=='a8c5df76e18c83c280e1236c549671358ac7b7fd62292035335416d0c77527cc'
assert sha(N/'GPU_run.py')==sha(D/'GPU_run.py')and sha(N/'plan.py')==sha(D/'plan.py')
cpu=json.loads(subprocess.check_output([sys.executable,'-B',str(N/'check_CPU.py')],text=True));assert cpu['passed']and len(cpu['cases'])==14
cpu['source_SHA256']={p.name:sha(p)for p in N.iterdir()if p.suffix in ['.py','.cpp','.sh','.log']};(N/'local-CPU-recording.json').write_text(json.dumps(cpu,indent=2)+'\n',encoding='utf8')
assert b'\r'not in(N/'CPU_inside.sh').read_bytes()
card=json.loads((D/'SOURCE_CARD.json').read_bytes());card.update(version='immutable-v2',original_source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'),original_CPU782={'actual':'910407.782','state':'FAILED1 retained','receipt_SHA256':'4c085f66411eb9b7449abb4506528313b11dd9db4fa7882f0f8443f302b9bfa6','subscope':'1HOSTcompile/link/currentheader/ldd-r/nm successful; finalmetadata directpathparser KeyError'},changes_only=['A1/B8/C8 heldoutputs+vectorstorage released beforefinaldrain andwall endpoint','metadata supportsactual782ldd directabsoluteandarrow lines, actualoldlog replay locally/CPU'],Native_GPU_math_plan_unchanged=True,output_lifetime='Caller-provided guarded512output survives aftertimer inallarms; allinternal temporaryMoE outputsandheldvectorstorage released inside timedregion, thenfinaldrain. Input/H2D/canary setup unchangedoutsideclock.',compile_reason='Only heldtemporarylifetime timing change requires1HOST recompile; metadataalone wouldnot justifyrecompile.')
(N/'SOURCE_CARD.json').write_text(json.dumps(card,indent=2)+'\n',encoding='utf8')
oldmf=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());members=['adapter.cpp','GPU_run.py','plan.py','check_CPU.py','CPU_build.py','CPU_inside.sh','invoke_CPU_once.py','original782-ldd-r.log','local-CPU-recording.json','SOURCE_CARD.json']
mf=oldmf|dict(files_SHA256={n:sha(N/n)for n in members},original_source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'),version='immutable-v2',GPU_GO=False)
(N/'SOURCE_MANIFEST.json').write_text(json.dumps(mf,indent=2)+'\n',encoding='utf8')
t=json.loads((G/'coordination/tasks/Kimi-MoE-prefill512-micro64-r391.json').read_bytes());t.update(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='IMMUTABLE_V2_CPU_SOURCE_READY',source='code/bw1000/Kimi-MoE-prefill512-micro64-r391/v2',source_manifest_SHA256=sha(N/'SOURCE_MANIFEST.json'),original_CPU782=card['original_CPU782'],CPU_submitted=False,GPU_submitted=False,GPU_GO=False,CPU_scope='Only1HOST timingfix recompile/link + sameactual14casePythonpayloadrecording; 0Native/GPUmathchange/noModel/noRuntime',local_CPU_recording_SHA256=sha(N/'local-CPU-recording.json'))
task=G/'coordination/tasks/Kimi-MoE-prefill512-micro64-r391.json';task.write_text(json.dumps(t,indent=2)+'\n',encoding='utf8')
paths=[D/'prepare_v2.py',D/'freeze_v2.py',*N.iterdir(),E/'Kimi391-CPU782-original-FAILED.json',E/'Kimi391-CPU782-compiled-subscope-and-v2-fix.json',task];assert all(p.is_file()for p in paths)
(L/'Kimi391-v2-stage-paths.json').write_text(json.dumps([str(p.relative_to(G)).replace('\\','/')for p in paths],indent=2)+'\n',encoding='utf8')
print(json.dumps({'passed':True,'MF':sha(N/'SOURCE_MANIFEST.json'),'files':len(paths),'HOST_TUs':1,'GPU_GO':False}))
