"""Freeze a concrete CPU-first card. No GPU admission or allocation claim yet."""
from pathlib import Path
import json,hashlib,datetime,statistics
D=Path(__file__).resolve().parent;G=D.parents[2];E=G/'results/bw1000/2026-11-02';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
prov=E/'GLM367-current038d-provenance.json';raw=json.loads(prov.read_bytes());src=E/'GLM367-current-dispatch-source-read.json';sources=json.loads(src.read_bytes())
vector=G/'evidence/bw1000/GLM-vector-component-result-r8539.json';old=json.loads(vector.read_bytes());m1=next(x for x in old['summary']if x['tokens']==1)
trace=G/'results/bw1000/2026-10-28/glm787-C1-profile-offline-recovered.json';profile=json.loads(trace.read_bytes())['data']
detail={}
for o in profile['owners']:
 for name,value in o['representative_detail_inclusive_stream_interval_ms_medians'].items():
  if name.endswith('.FFN'):detail.setdefault(name,[]).append(value)
current={p:h for p,h in raw['receipts'][0]['reused_objects_SHA256'].items()if any(x in p for x in ['vector-0.o','routed.o','row2','dispatch.o'])}
q=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),decision='ONE_CONCRETE_M1_GATE_ONLY_CANDIDATE_CPU_FIRST',GPU=False,Native_dlopen=False,weights_opened=False,
 current_Native_SHA256='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00',SDK=sources['SDK_receipt'],current_objects_SHA256=current,
 source_FFN_SHA256='c716b37fd3fe786cece00a5b4a309d9eb13b76e4bbdb8cc8843be936c708e64b',dispatch9284_SHA256='2bedb52f033e73cdc68addaa6c3ec893802c8e3e36980a840b397083bddd3ebf',
 actual_shapes=dict(Native_row_tokens=1,routes=8,hidden=6144,experts=256,topk=8,TP=8,local_intermediate=256,expert_global_intermediate=2048,shared_global_intermediate=2048,gate_up_weight_per_TP_expert=[256,6144],down_weight_per_TP_expert=[6144,256],router_weight=[256,6144],weight_and_activation_format='FP8 E4M3 bytes with F32 block128 scales; BF16 gate/up/Silu/Mul/down outputs and F32 routed weighting/merge/TP'),
 current_path='Current vector policy OFF in361; even ON selects only routes32/64/128. M1gate/down call same f359 grouped fallback -> original custom scalar kernels; not per-expert Torch GEMM. SharedFFN uses SDK Fp8BlockLinear.',
 candidate='Independent defaultOFF M1 gate-only bit; routes8 gate/up calls already-linked8528 widefetch preserving original W64 FP32 tree/scales; down remains current scalar. All other geometries/9284 row2/Frame ABI8 preserved. No new GPU TU.',
 previous_M1_component=m1,previous_M1_boundary=old['scope'],
 existing_Model_cost_basis=dict(path=str(trace.relative_to(G)).replace('\\','/'),SHA256=sha(trace),actual_GPU_step=profile['actual_GPU_step'],C1_T1_samples_per_owner=3,normal_owners=len(profile['owners']),representative_FFN_enclosure_ms_across_owner_medians={k:dict(min=min(v),median=statistics.median(v),max=max(v))for k,v in detail.items()},boundary='Actual earlier Model C1T1 inclusive same-stream FFN envelopes establish nonzero FFN cost; PP2 geometry/short history, not current361 PP4 cost or exclusive GPUbusy. No multiplication by layers and no Model speedup projection.'),
 current361_M1_FFN_exclusive_cost='UNKNOWN; original361 captures wholeNative calls without FFN subevents',
 complete_component_entry='Existing current038d exported glm_layer_*r5670 layer38 TP8 API; whole layer wall includes complete FFN/router/shared/TP/drain. No new replay framework. GPU feasibility still conditional on CPU close and first actual constructor success.',
 planned_component_fixture='Real layer38 checkpoint subset, synthetic replicated BF16 hidden and synthetic prefix512, actual M1 decode; inputs changed between blocks, state restored identically, canary/finite/independent repeat/full output comparison and normal8owner destroy; component not actual Model activation/quality qualification',
 excluded=['341 activation/downquant fusion zero original retainers','RMS descriptor cache already present','bitdecode same math old negative and existing hardware ldexp','BF16 unpack/cache stale large-U lowreuse negatives','8528 down-vector M1 not adopted','schedule interleave frequency grid'],
 fallback='If complete current Runtime layer entry cannot close without new architecture or has no measurable full-wall effect, stop this card; alternative router F32GEMV+sigmoid/bias/top8 normalization fusion requires separate complete FFN evidence, not launched',
 raw_read_artifacts={str(p.relative_to(G)).replace('\\','/'):sha(p)for p in [prov,src]},raw_reads_published=False)
out=E/'GLM367-first-source-shape-cost-card.json';out.write_text(json.dumps(q,indent=2)+'\n',encoding='utf8')
members=['baseline_dispatch_r9284.cpp','dispatch_M1_r367.cpp','glm_vector_dispatch_r8540.hpp','check_dispatch_CPU.cpp','CPU_build.py','CPU_inside.sh','prepare_dispatch.py','prepare_CPU_claim.py']
for name in ['CPU_inside.sh']:
 p=D/name;p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'));assert b'\r'not in p.read_bytes()and p.read_bytes().endswith(b'\n')
for name in ['CPU_build.py','prepare_dispatch.py','prepare_CPU_claim.py']:compile((D/name).read_bytes(),name,'exec')
mf=D/'CPU_SOURCE_MANIFEST.json';mf.write_text(json.dumps(dict(schema=1,scope='367 onehostdispatchTU CPU-only, noGPUGO',files_SHA256={n:sha(D/n)for n in members}),indent=2)+'\n',encoding='utf8')
claim=dict(schema=1,UTC=q['UTC'],owner='/root/glm_operator_tuning',state='CPU_READY_ONE_M1_GATE_ONLY_CURRENT038d_DISPATCH',source=str(D.relative_to(G)).replace('\\','/'),CPU_SOURCE_MANIFEST_SHA256=sha(mf),CPU_claim=dict(parent=910407,node='a14r2n20',CPU=4,memory_MiB=8192,Slurm_seconds=900,inside_seconds=840,Native_dlopen=False,GPU=False,weights_opened=False),GPU_GO=False,GPU_submitted=False,GPU_claim=None,planned_conditional_GPU='Only after actualCPU/sourcebinding and ordinarypublished preciseclaim, fresh910371/a02r3n03,8TP samecomponent owners,eachmeasurement<=600s. No a02r3n07 secondmechanism established.',mechanism=q['candidate'],first_card=str(out.relative_to(G)).replace('\\','/'),first_card_SHA256=sha(out),current_Native_SHA256=q['current_Native_SHA256'],new_GPU_TUs=0,CPU_test='Actual compiled selector with recordingprovider stubs;832 geometry/flag combinations+nestedactive/meta/error/device isolation; no GPU numeric qualification',Model_gain='UNKNOWN',no_gpu_occupation_during_preparation=True)
(G/'coordination/tasks/GLM-smallM-FFN-r367.json').write_text(json.dumps(claim,indent=2)+'\n',encoding='utf8')
print(json.dumps({'manifest_SHA256':sha(mf),'first_card_SHA256':sha(out),'Model_cost_enclosures':q['existing_Model_cost_basis']['representative_FFN_enclosure_ms_across_owner_medians'],'GPU_GO':False}))
