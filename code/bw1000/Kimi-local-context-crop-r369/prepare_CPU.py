from pathlib import Path
import json,hashlib,ast,base64
D=Path(__file__).resolve().parent;G=D.parents[2];L=G.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=['private_attention.py','query_window_r369.py','check_CPU.py','CPU_main.py','CPU_inside.sh']
for name in files:
 p=D/name
 if p.suffix=='.py':ast.parse(p.read_text())
 if p.suffix=='.sh':assert b'\r'not in p.read_bytes()
T=G/'code/bw1000/kimi-dflash2-trained-worker';Q=G/'code/bw1000/Kimi-long-draft-RPC-CPU-r333'
trained=json.loads((T/'DFlash2-trained-manifest-r9019.json').read_bytes());deps={}
for name in ['dflash2_creator_factory_r9007.py','dflash2_strict_loader_r9007.py','r9000_torchspec_deepseek_eagle.py','r9000_torchspec_dflash.py','r9000_torchspec_dflash2.py','r9000_torchspec_dspark.py','r9000_torchspec_llama3_eagle.py']:
 assert sha(T/name)==trained['incoming_source_SHA256'][name]
 deps['/refactor/incoming/'+name]=sha(T/name)
deps['/refactor/incoming/config.json']=trained['incoming_source_SHA256']['config.json']
deps['/refactor/incoming/Kimi-long-draft-RPC-CPU-r333/query_long_projected_r323.py']=sha(Q/'query_long_projected_r323.py')
proof=dict(scope='Actual pinned strict-loader factory and selected five-layer implementation, no GPU or checkpoint',dependencies_SHA256=deps,
 K_V_dependency='K3DSparkMLAAttention.forward: Q only draft_hidden; K/V from tokenwise Linear(cat(context,draft))->split->RMSNorm(mean -1 only)->Linear; context rows never depend on anchor/draft/query. DFlash2DecoderLayer convolves only eight draft rows before/after attention; context never updated between layers.',
 position_dependency='Pinned config YaRN factor32/theta50000/original32768/mscale1/all_dim1. inv_freq and mscale depend only fixed configuration; t is absolute index. Attention still requests a sliced RoPE table of ctx_len+8. Crop alone fails absolute dp/cp indexing; private AST forward changes only seq_len=absolute_span (static integer original L+8).',
 mask_algebra='dp=L+j,j0..7;cp=k,k0..L-1. abs(L+j-k)<4096 iff k>=L+j-4095. Union over all8 queries k>=L-4095. Keep context suffix max(0,L-4095)..L-1, preserve per-query original mask columns and absolute cp/dp. All8 draft self-key columns remain (max distance7). Fifth global layer uses fullctx/fullmask.',
 condition='C1 contiguous positions0..L-1,8 fixed draft positions,L<=32760; no padding/crossrow generalization',
 nonchanges='No weight/head/projection/selector math changes; original five class files untouched; no persistent KV lifecycle, cache or new allocator. Default crop_localFalse. Local borrowed-weight copies do not mutate original model modules.',
 cost_bound='Root368 measured near32638 queryRPC1.4829100701s / first-last4.9121621270s. Holding other costs/acceptance fixed, >15tok/s requires >74.9986% query saving; even zeroquery insufficient for this first-request33card per-card generation gain. No speed/acceptance prediction.')
(D/'DEPENDENCY_PROOF.json').write_text(json.dumps(proof,indent=2)+'\n')
files.append('DEPENDENCY_PROOF.json')
m=dict(files_SHA256={n:sha(D/n)for n in files},dependencies_SHA256=deps,default_crop_enabled=False,GPU_GO=False,GPU=0,CPU_claim=dict(parent=910407,node='a14r2n20',CPUs=4,memory_MiB=8192,hard_s=600,inner_s=570),actual_CPU_scope='Tiny actual pinned five-layer Torch factory and original333 query, seven lengths1/1024/4095/4096/8635/32638/32760; purecrop IndexError negative plus absolute-span candidate hidden/logits finite/NRMSE/proposals diagnostic and immutableweights/input/module routing. No fullCPU checkpoint or Native.')
(D/'SOURCE_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');h=sha(D/'SOURCE_MANIFEST.json')
task=dict(owner='/root/kimi_operator_tuning',state='SOURCE_DEPENDENCY_PROOF_CPU_PENDING',source=str(D.relative_to(G)).replace('\\','/'),source_manifest_SHA256=h,GPU_GO=False,new_GPU=0,CPU_claim=m['CPU_claim'],CPU_scope=m['actual_CPU_scope'],next_possible_GPU='One draft only910371/a02r3n15<=600s, CPUkeeper<=900; only after actual CPU/source/full query plan/precise claim ordinarypush, no targetModel/no windowgrid',mathematical_delta='Remove only permanently masked context rows in first4layers; preserve static absolute RoPE spanL+8; global fifth unchanged',Model_gain='UNKNOWN; root368 strict15/percard bounds retained')
(G/'coordination/tasks/Kimi-local-context-crop-r369.json').write_text(json.dumps(task,indent=2)+'\n')
payload={n:base64.b64encode((D/n).read_bytes()).decode()for n in files+['SOURCE_MANIFEST.json']}
script="""from pathlib import Path
import base64,json,hashlib,subprocess,os
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-local-context-crop-r369';assert not D.exists();D.mkdir()
for n,b in __PAYLOAD__.items():(D/n).write_bytes(base64.b64decode(b))
assert hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()==__HASH__
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','--gres=none','--mem=8192M','-w','a14r2n20','-t','00:10:00','--job-name=Kimi-r369-whole-CPU','--output='+str(R/'logs/Kimi-r369-whole-CPU-%J.log'),'--export=ALL','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
env=dict(os.environ,WAVE_INSIDE_R2323=D.name+'/CPU_inside.sh')
with(R/'logs/Kimi-r369-whole-CPU-client.log').open('x')as f:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(CPU_submitted=True,GPU=0,PID=p.pid,label='Kimi-r369-whole-CPU',source_manifest_SHA256=__HASH__)))
""".replace('__PAYLOAD__',repr(payload)).replace('__HASH__',repr(h))
ast.parse(script);(L/'remote369-upload-CPU.py').write_text(script,encoding='utf8')
print(json.dumps(dict(source_manifest_SHA256=h,files=files,CPU_GPU=0,helper_AST_passed=True)))
