import pathlib,hashlib,json,base64,ast
repo=pathlib.Path(r'C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test');s=repo/'code/bw1000/Deep-operator-search-r340';L=repo.parent
for p in s.glob('*.py'):ast.parse(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'method':'Custom BF16 RMSNorm/block sweep and intermediate-BF16-preserving tail64 RoPE fusion','scope':'same-shape synthetic component, custom unfused baseline; not actual activation or production speed claim','files':{p.name:sha(p)for p in s.iterdir()if p.is_file()},'dependencies':['/opt/dtk HIP compiler/runtime gfx936','container torch ROCm (exact versions and shared-object SHA collected on run)'],'model_shapes':{'compressor':[128,512],'hidden':4096,'tail_rope':64},'quality':'float32 PyTorch reference, BF16 intermediate, finite required and NRMSE<0.01; maxabs retained','pairs':'same GPU ABBA x5, 100 calls/interval; launch/event/complete wall and baseline drift retained','source_basis':'Deep-C128-weight-cache-r172/src/deepseek_v4_compressor_fp32_c128_r19.cpp lines243/244; no cached-weight route','sealed_negatives_read':'Deep-C1-1195-diagnosis-r172, Deep1221-current-Native-and-emitter-cost-r151; excluded smallT/cache/readback/bankcopy','GPU_budget':{'parent':'910374','nodes':['a02r4n04','a02r4n05'],'max_cards':16,'seconds_per_GPU':600,'batch_hard_seconds':1800},'CPU_budget':{'parent':'910407','node':'a14r2n20','CPUs':4,'memory_MiB':8192,'seconds':600}}
(s/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
task=dict(owner='bw1000-root/deep_operator_tuning',state='SOURCE_CPU_BUILD_PENDING_ROOT_PUBLICATION',method=manifest['method'],source=str(s.relative_to(repo)).replace('\\','/'),manifest_SHA256=sha(s/'manifest.json'),resources=manifest['GPU_budget'],CPU=manifest['CPU_budget'],no_shared_dependencies_mutated=True,no_model_load=True,once_only=True)
(repo/'coordination/tasks/Deep-operator-search-r340.json').write_text(json.dumps(task,indent=2)+'\n')
files={p.name:base64.b64encode(p.read_bytes()).decode() for p in s.iterdir()if p.is_file()}
remote="""import pathlib,base64,json,subprocess,os
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014'); S=R/'incoming/Deep-operator-search-r340';S.mkdir(exist_ok=True)
FILES=FILES_PLACEHOLDER
for n,b in FILES.items():
 p=S/n;v=base64.b64decode(b)
 if p.exists():assert p.read_bytes()==v,n
 else:p.write_bytes(v)
env=dict(os.environ)
for k in list(env):
 if k.startswith(('SLURM_','PMI_','PMIX_','OMPI_'))and k not in ['SLURM_CONF','SLURM_CONF_SERVER']:env.pop(k)
a=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','-w','a14r2n20','--gres=none','--mem=8192M','-t','00:10:00','--job-name=Deep340-CPU','--export=ALL,WAVE_INSIDE_R2323=Deep-operator-search-r340/build_inside.sh','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
f=(R/'logs/Deep340-CPU-client.log').open('x');p=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'PID':p.pid,'argv':a,'log':str(R/'logs/Deep340-CPU-client.log')}))
""".replace('FILES_PLACEHOLDER',repr(files))
(L/'Deep340-stage-remote.py').write_text(remote,encoding='utf8')
print(json.dumps({'manifest_sha':sha(s/'manifest.json'),'CPU_python_ast':True,'files':list(files)}))
