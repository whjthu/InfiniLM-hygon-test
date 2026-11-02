from pathlib import Path
import json,hashlib,ast,base64
D=Path(__file__).resolve().parent;G=D.parents[2];L=G.parent;S=D.parent/'Kimi-local-window-first-divergence-r371'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());deps=dict(parent['dependencies_SHA256'])
runtime=D.parent/'Kimi-long-draft-RPC-CPU-r333';rm=json.loads((runtime/'SOURCE_MANIFEST.json').read_bytes())
deps.update({'/refactor/incoming/Kimi-long-draft-RPC-CPU-r333/'+n:h for n,h in rm['files_SHA256'].items()})
deps['/refactor/incoming/Kimi-long-draft-RPC-CPU-r333/SOURCE_MANIFEST.json']=sha(runtime/'SOURCE_MANIFEST.json')
Lctx=32638;size=5*Lctx*(576+16384)*2
design=dict(scope='Private preRoPE KV_A/KV_B CPU prototype; originalattention.forward/fullmask/KVnorm/RoPE/SDPA/Q/O/head/selector unchanged; no GPU/target/checkpoint load',review161addendum=dict(path='analysis/bw1000/reviews/astra-review-161-kimi-cache-cost-addendum.json',SHA256=sha(G/'analysis/bw1000/reviews/astra-review-161-kimi-cache-cost-addendum.json')),minimum_change='Private shallowmodel/layer/attention routing copies borrow originalparameters/buffers. TwoLinear wrappers firstcompute fullM=L+8 andharvest prefixviews; latercompute originalLinear onlysuffixdelta+8 andcat originalfullshape. KVnorm staysfulltokenwise. cacheflag defaultFalse.',lifetime='Contextowner/epoch/request/modelinstance/weighttag/KVweightmodule/deviceptr/dtype/version/epsilon signature/cached_length. Frozeninferenceweights mayhave noversion, nohotreplacement or perqueryfullweightSHA.',publication='Stage10A/B values across5layers; originalcodec4GiBfence mustsuccess beforecommitting allpending. PrivateSession delegates original333dispatch andonly addsabort/drop on failure; rejected/partialfailure cannotpublish. append remains original. retire/close/newEpoch drop; gap>8 legal cumulativeappend executesoriginalfullquery refresh, ARtail original.',memory=dict(L=Lctx,perlayer_A=[1,Lctx,576],perlayer_B=[1,Lctx,16384],dtype='BF16',all5_cache_bytes=size,all5_cache_GiB=size/(1<<30),capacity32760_bytes=5*32760*16960*2,fullM_view_extra8rows_bytes=5*8*16960*2,old_plus_next_pending_bytes=size+5*(Lctx+8)*16960*2,delta0_reuses_prior_cache_reference=True,meaning='Shape arithmetic not actualGPUpeak; fullM cat/finite/attention temporaries and originalfloor mustbe measured later.'),CPU_scope='Actual pinnedtinyfactory+actual333 Session/LongCommittedFeatureContext/originalquery. BF16actualwire andF32reference decodedBF16->F32 only. Threecontext513/8635/32638; cold+8queries/sevenaccepted8, fullOFFvsON stage/hidden/logits/proposals. Additionaldelta0/K1/partial1,3,8/gap9/owner/epoch/version/rejectedcommit/fence/projection/nonfinite/closeerror/tail cases. No repeat740 productionwidthmatrix run.',timing='Recordmodel/head CPUstartup, allpromptappend/seal/query/acceptedappend/retire/finalclose, perLinear compute/cat hostwall,logical/unique cachebytes, peakRSS. CPUquerytiming includes traceclones andvalidation; not GPUperformance. Futuredevicecard mustkeep fullcold+8cycle+startup andRPC/dispatch scope clear.',cost_boundary='Originalnear firstRPC.592191957 retained; later7zero conditionalbest14.174tok/s/+22.15% is an optimisticbound, not predictedbenefit; below15 doesnotkillpartialimprovement. No cache+warmup mixedgain.',quality='Reportfinite/NRMSE/maxabs/proposals; no tokenbitwise requirement or Modelquality/acceptance certification. FirstfullM mathpreserved; delta+8 shape variation needs actualBF16device qualification later.')
(D/'DESIGN.json').write_text(json.dumps(design,indent=2)+'\n')
files=['prefix_cache.py','check_CPU.py','CPU_main.py','CPU_inside.sh','DESIGN.json']
for n in files:
 p=D/n
 if p.suffix=='.py':ast.parse(p.read_text())
 if p.suffix=='.sh':assert b'\r'not in p.read_bytes()
m=dict(files_SHA256={n:sha(D/n)for n in files},dependencies_SHA256=deps,GPU_GO=False,GPU=0,CPU_claim=dict(parent=910407,node='a14r2n20',tasks=1,CPUs=4,memory_MiB=8192,hard_s=600,inner_s=570),scope=design['CPU_scope'],engineering_bound='<=90min firstengineering verdict, not completion guarantee; if no executableclosure stopwith exactgap, no broadcache framework')
(D/'SOURCE_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');h=sha(D/'SOURCE_MANIFEST.json')
t=dict(owner='/root/kimi_operator_tuning',state='PRIVATE_PRE_ROPE_TWO_LINEAR_SOURCE_CPU_PENDING',source=str(D.relative_to(G)).replace('\\','/'),source_manifest_SHA256=h,GPU_GO=False,new_GPU=0,target_cards=0,CPU_claim=m['CPU_claim'],CPU_scope=m['scope'],review='161cacheaddendum privatepreRoPEfirst, unchangedattention/fullmask/RoPE, defaultOFF',no_shared_model_parameter_buffer_modification=True,no_target_or_690_697_replay=True,next='OnlyafteractualCPU/source results a separateconcrete single-draft deviceclaim may be considered; no GPUauthorized by this CPUcard',Model_gain='UNKNOWN')
(G/'coordination/tasks/Kimi-preRoPE-prefix-cache-r376.json').write_text(json.dumps(t,indent=2)+'\n')
payload={n:base64.b64encode((D/n).read_bytes()).decode()for n in files+['SOURCE_MANIFEST.json']}
script="""from pathlib import Path
import json,hashlib,base64,subprocess,os
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-preRoPE-prefix-cache-r376';assert not D.exists();D.mkdir()
for n,b in __PAYLOAD__.items():(D/n).write_bytes(base64.b64decode(b))
assert hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()==__HASH__
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','--gres=none','--mem=8192M','-w','a14r2n20','-t','00:10:00','--job-name=Kimi-r376-prefix-cache-CPU','--output='+str(R/'logs/Kimi-r376-prefix-cache-CPU-%J.log'),'--export=ALL','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
env=dict(os.environ,WAVE_INSIDE_R2323=D.name+'/CPU_inside.sh')
with(R/'logs/Kimi-r376-prefix-cache-CPU-client.log').open('x')as f:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(CPU_submitted=True,GPU=0,PID=p.pid,label='Kimi-r376-prefix-cache-CPU',source_manifest_SHA256=__HASH__)))
""".replace('__PAYLOAD__',repr(payload)).replace('__HASH__',repr(h))
ast.parse(script);(L/'remote376-upload-CPU.py').write_bytes(script.encode())
print(json.dumps(dict(source_manifest_SHA256=h,GPU=0,helper_AST_passed=True,cache_preRoPE_GiB=size/(1<<30))))
