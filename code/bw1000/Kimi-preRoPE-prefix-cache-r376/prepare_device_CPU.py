from pathlib import Path
import hashlib,json,ast,base64
B=Path(__file__).resolve().parent;D=B/'device-v1';G=B.parents[2];L=G.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
core=json.loads((B/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(B/n)==h for n,h in core['files_SHA256'].items())
deps=dict(core['dependencies_SHA256']);deps.update({'/refactor/incoming/Kimi-preRoPE-prefix-cache-r376/'+n:h for n,h in core['files_SHA256'].items()});deps['/refactor/incoming/Kimi-preRoPE-prefix-cache-r376/SOURCE_MANIFEST.json']=sha(B/'SOURCE_MANIFEST.json')
deps['/refactor/candidates/Kimi-preRoPE-prefix-cache-r376/CPU-result.json']='919c8faf461b4536a509ee76c2f52f454d6bbfac81389454bc4f1a14c2dc7bda'
plan=dict(default_GPU_GO=False,geometries=[1024,8635,32638],per_geometry_sequence_order=['warm_OFF','warm_ON','formal_OFF','formal_ON','formal_ON','formal_OFF'],freshContext_each_sequence=True,queries_per_sequence=8,synthetic_committed8_frames_per_sequence=7,total_sequences=18,warm_queries=48,formal_queries=96,total_queries=144,prompt_append_frames=498,committed_append_frames=126,begin=18,seal=18,retire=18,close=18,protocol_commands=840,original_codec_fences=912,GPU_if_separately_claimed=dict(cards=1,node='a02r3n15',parent=910371,CPUs=8,memory_MiB=65536,hard_s=600,CPU_keeper_parent=910407,CPU_keeper_node='a14r2n20',CPU_keeper_hard_s=900,target_cards=0),timing='Primarycontinuouschecked8cycle includes completequeries+acceptedappends+pending/cat/fence andinline canary/stage readback/numeric/save overhead. Dispatchsum separate, not pureGPU. Prefixbegin/append/seal, allwarm sequences, strictload/startup, retireclose andallcardseconds separately. Firstprocess/shapecalls are not proven physicalGPUcold.',stage_scope='Everyquery: all5layers KV_A/KVnorm/KV_B/O fixed3contextrow+8draft samples; full8hidden and7xvocablogits compared, actual7proposals preserved. New cache fullcomputed outputsfinitecheck stays insidecachedquery. No rawactivationserialization.',numeric=dict(alert=.02,investigation_stop=.10,meaning='Bothlocaldiagnostic thresholds, not globalquality acceptability; finitehidden/logit>.10 savesallfields beforestop. Semantic/state/canary/invalidproposal/nonfinite stop. Old369.02FAILunchanged. No actualtargetacceptance.'),CPU_scope='Exact shared plan/run_plan/run_sequence using actualtinyBF16 factory+333 Session+frozen752cache;18contexts/144queries/840commands/912fence counts. Deliberate invalidreference negative verifies saved fullstage/hidden/logit/props/state before investigationstop. No strictloader/checkpoint/device/floor qualification.',warmup_independent='No independentstartupwarmup/overlap mechanism mixedwithcache; pergeometrywarm sequences mandatedforpairedmeasurement andfullyaccounted. No target.')
(D/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
files=['sequence_consumer.py','GPU_driver.py','check_plan_CPU.py','CPU_main.py','CPU_inside.sh','PLAN.json']
for n in files:
 p=D/n
 if p.suffix=='.py':ast.parse(p.read_text())
 if p.suffix=='.sh':assert b'\r'not in p.read_bytes()
m=dict(files_SHA256={n:sha(D/n)for n in files},dependencies_SHA256=deps,GPU_GO=False,GPU=0,CPU_claim=dict(parent=910407,node='a14r2n20',tasks=1,CPUs=4,memory_MiB=8192,hard_s=600,inner_s=570),scope=plan['CPU_scope'])
(D/'CPU_SOURCE_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');h=sha(D/'CPU_SOURCE_MANIFEST.json')
tp=G/'coordination/tasks/Kimi-preRoPE-prefix-cache-r376.json';t=json.loads(tp.read_bytes());t.update(state='CPU752_PASS_EXACT18SEQUENCE_DRIVER_CPU_PENDING',GPU_GO=False,new_GPU=0,device_CPU_source_manifest_SHA256=h,device_CPU_claim=m['CPU_claim'],device_CPU_scope=m['scope'],exact_future_plan=plan);tp.write_text(json.dumps(t,indent=2)+'\n')
payload={n:base64.b64encode((D/n).read_bytes()).decode()for n in files+['CPU_SOURCE_MANIFEST.json']}
script="""from pathlib import Path
import json,hashlib,base64,subprocess,os
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');B=R/'incoming/Kimi-preRoPE-prefix-cache-r376';D=B/'device-v1';assert not D.exists();D.mkdir()
for n,b in __PAYLOAD__.items():(D/n).write_bytes(base64.b64decode(b))
assert hashlib.sha256((D/'CPU_SOURCE_MANIFEST.json').read_bytes()).hexdigest()==__HASH__
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','--gres=none','--mem=8192M','-w','a14r2n20','-t','00:10:00','--job-name=Kimi-r376-device-plan-CPU','--output='+str(R/'logs/Kimi-r376-device-plan-CPU-%J.log'),'--export=ALL','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
env=dict(os.environ,WAVE_INSIDE_R2323='Kimi-preRoPE-prefix-cache-r376/device-v1/CPU_inside.sh')
with(R/'logs/Kimi-r376-device-plan-CPU-client.log').open('x')as f:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(CPU_submitted=True,GPU=0,PID=p.pid,label='Kimi-r376-device-plan-CPU',source_manifest_SHA256=__HASH__)))
""".replace('__PAYLOAD__',repr(payload)).replace('__HASH__',repr(h))
ast.parse(script);(L/'remote376-device-upload-CPU.py').write_bytes(script.encode())
print(json.dumps(dict(CPU_manifest_SHA256=h,GPU=0,helper_AST_passed=True,exact_queries=144,commands=840,fences=912)))
