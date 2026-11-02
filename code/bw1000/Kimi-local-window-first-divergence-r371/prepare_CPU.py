from pathlib import Path
import json,hashlib,ast,base64
D=Path(__file__).resolve().parent;G=D.parents[2];L=G.parent;B=D.parent/'Kimi-local-context-crop-r369'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original=json.loads((B/'gpu-v1/GPU_MANIFEST.json').read_bytes())
assert all(sha(B/'gpu-v1'/n)==h for n,h in original['files_SHA256'].items())
deps=dict(original['dependencies_SHA256'])
deps.update({'/refactor/incoming/Kimi-local-context-crop-r369/gpu-v1/'+n:h for n,h in original['files_SHA256'].items()})
deps['/refactor/incoming/Kimi-local-context-crop-r369/gpu-v1/GPU_MANIFEST.json']=sha(B/'gpu-v1/GPU_MANIFEST.json')
Lctx=32638;start=Lctx-4095;counts=[]
for j in range(8):
 full=[abs(Lctx+j-k)<4096 for k in range(Lctx+8)]
 assert not any(full[:start])and all(full[Lctx:]);counts.append(sum(full[:Lctx]))
proof=dict(state='SOURCE_ONLY_NO_NEW_GPU',actual_failed_report='results/bw1000/2026-11-02/Kimi369-GPU690-query-FAIL-sealed.json',failed_SHA256=sha((G/'results/bw1000/2026-11-02/Kimi369-GPU690-query-FAIL-sealed.json')),missing_690_raw='Not captured; no fabricated reconstruction. This CPU uses random parameters and a distinct geometry sample.',frozen369_GPU_manifest_SHA256=sha(B/'gpu-v1/GPU_MANIFEST.json'),source_semantics=dict(mask=dict(L=Lctx,keep_cp=[start,Lctx-1],dp=[Lctx,Lctx+7],retained_context_union_rows=4095,valid_context_rows_per_query=counts,all8draft_columns_retained=True),RoPE='Both paths call the same original rotary module; bothrequest absoluteL+8=32646. Original full total_len32646 and crop total_len4103 affect K/V reshape, but privateforward only replaces rotary seq_len. Cacheddtype depends on loader-restored defaultdtype; actual690bufferdtype was not logged, cannot infer it from CPU. Both arm bufferobject/dtype remain shared by source.',copy='copy.copy(layer) copies Python routing only, _modules dictionary copied; SpanAttention.original references originalself_attn. No parameters/buffers reconstructed or cast. Original function globals copied, then only call wrapper captures exactQKV and returns originalSDPA unchanged.',dependency='KVA/latentRMS/KVB tokenwise; normalization mean(-1), never over context. Draft groupedconv only8draft, context neverupdated. Globalfifth unchanged.',mathematical_conclusion='No real-number mask/absoluteposition/norm/borrowedbuffer semantic difference identified. Finiteprecision/backend/M-shape changes remain hypothesis, not established cause.'),hypotheses=[dict(name='Retained KV recomputation changes first',test='Same captured firstattention draft input/context/weights. Compare ALL retained4095context+8draft rows at Q,KV_A,latentRMS,KV_B; report firstnonzero/finite/NRMSE/maxabs/RMS.'),dict(name='Same-QKV masked attention changes with shape',test='Materialize fullfirstattention Q,K,V once. Use samevalues; crop only allFalse columns, preserve perquery mask, contiguousK/V as normal candidate. Projection absent. Compare rawSDPAoutput beforeOproj.'),dict(name='Full replay state/input differs',test='Original completeOFFquery capture8xH attentioninput/output; samefirstattention fullreplay output vs original validates replaypath. Retained cache/bufferSHA and weights/module identity guard, no prefixreload.')],bounds=dict(CPU_max_s=600,CPU_cores=4,CPU_memory_MiB=8192,CPU_scope='Randomparameters, fullproduction KV_A hidden7168/output576/M32646 vs4103; latent512 and full16384 KV_B output only12alignedrows; actualtiny fullquery69commands + samefirstlayer producer/fixedQKV controls. No checkpointloaded/GPU/native.',next_GPU='NOT_YET_CLAIMED: 1draftn15,600s incl strictload/CPU900. Original fullquery exactlyonce + onefullfirstattention replay + onefixedQKV cropSDPA + onecropfirstattention replay. Scalar summaries saved before any diagnostic stop; no full5layerON/ABBA/target.',GPU_reference_QK_V_bytes=64*(Lctx+8)*(192+128)*2,GPU_firstproducer_retained_KVB_bytes=4103*16384*2,CPU_expected_peak_below_claim='Estimated <4GiB for described tensors/randomattention; actualru_maxrss recorded; not GPU allocator peak',no_unbounded_activation_logging=True,original_numeric_threshold=0.02),cost_bound='r368 unchanged: >74.9986% queryRPC saving needed strict15near ifallothercosts/acceptance held; zeroquery still insufficientfirstrequest33card percard. No speed/acceptance forecast.')
(D/'SOURCE_REVIEW.json').write_text(json.dumps(proof,indent=2)+'\n')
names=['first_layer.py','check_CPU.py','CPU_main.py','CPU_inside.sh','SOURCE_REVIEW.json']
for n in names:
 p=D/n
 if p.suffix=='.py':ast.parse(p.read_text())
 if p.suffix=='.sh':assert b'\r'not in p.read_bytes()
m=dict(files_SHA256={n:sha(D/n)for n in names},dependencies_SHA256=deps,GPU_GO=False,GPU=0,CPU_claim=dict(parent=910407,node='a14r2n20',CPUs=4,memory_MiB=8192,hard_s=600,inner_s=570),scope=proof['bounds']['CPU_scope'],no_checkpoint_load=True,source_only_GPU_plan=proof['bounds']['next_GPU'])
(D/'SOURCE_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');h=sha(D/'SOURCE_MANIFEST.json')
t=dict(owner='/root/kimi_operator_tuning',state='SOURCE_CPU_FIRSTLAYER_CAUSAL_DIAGNOSTIC_PENDING',source=str(D.relative_to(G)).replace('\\','/'),source_manifest_SHA256=h,GPU_GO=False,new_GPU=0,CPU_claim=m['CPU_claim'],CPU_scope=m['scope'],review='161 accepted fixedfirstlayer KVrecompute vs sameQKV maskedattention split; no causalclaimfrom accumulated5layerNR',previous_trial='369GPU690 FAILED1/56s,missingfailedqueryraw preserved; no unchanged replay/thresholdrelax/target',next_GPU_plan=m['source_only_GPU_plan'],Model_gain='UNKNOWN')
(G/'coordination/tasks/Kimi-local-window-first-divergence-r371.json').write_text(json.dumps(t,indent=2)+'\n')
payload={n:base64.b64encode((D/n).read_bytes()).decode()for n in names+['SOURCE_MANIFEST.json']}
script="""from pathlib import Path
import json,hashlib,base64,subprocess,os
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-local-window-first-divergence-r371';assert not D.exists();D.mkdir()
for n,b in __PAYLOAD__.items():(D/n).write_bytes(base64.b64decode(b))
assert hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()==__HASH__
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','--gres=none','--mem=8192M','-w','a14r2n20','-t','00:10:00','--job-name=Kimi-r371-firstlayer-CPU','--output='+str(R/'logs/Kimi-r371-firstlayer-CPU-%J.log'),'--export=ALL','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
env=dict(os.environ,WAVE_INSIDE_R2323=D.name+'/CPU_inside.sh')
with(R/'logs/Kimi-r371-firstlayer-CPU-client.log').open('x')as f:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(CPU_submitted=True,GPU=0,PID=p.pid,label='Kimi-r371-firstlayer-CPU',source_manifest_SHA256=__HASH__)))
""".replace('__PAYLOAD__',repr(payload)).replace('__HASH__',repr(h))
ast.parse(script);(L/'remote371-upload-CPU.py').write_bytes(script.encode())
print(json.dumps(dict(source_manifest_SHA256=h,GPU=0,helper_AST_passed=True)))
