"""Check real Qwen launch interpretation against observed source/config bytes."""
from pathlib import Path,PurePosixPath
import ast,hashlib,json,subprocess,sys,time
D=Path(__file__).resolve().parent;F=D.parent/'hygon-training-integration'
sys.path.insert(0,str(F))
from tools.hygon_vlm_recipe import build_launch
from vlm.runtime.native_recipe_contract import native_recipe_spec
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
p=A/'cross_platform_reference_r1/raw/node26/node/results/qwen-cuda8-gc-on-v1-sdpa-r1/launch.json'
raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()=='28732596ca74bf491ee2c33f3ed8e5cbb2054e761195f3f49a68183b42ae7400'
plan=build_launch('T01.sft',raw,PurePosixPath('/runtime/swift'),PurePosixPath('/models'),PurePosixPath('/data/train.jsonl'),PurePosixPath('/plugins'),PurePosixPath('/diagnostic/train'),29501)
spec=native_recipe_spec(plan)
observed=json.loads((D/'retry-metadata-receipt.json').read_bytes())
for name,pin in observed['files'].items():
 data=(D/(name+'.original')).read_bytes();assert len(data)==pin['bytes'] and hashlib.sha256(data).hexdigest()==pin['SHA256']
config=json.loads((D/'config.original').read_bytes());assert config['architectures']==[spec['model_class']] and config['model_type']=='qwen3_vl'
index=json.loads((D/'index.original').read_bytes());assert len(index['weight_map'])==observed['tensors']==750
assert set(index['weight_map'].values())=={s['name'] for s in observed['shards']}
tree=ast.parse((D/'model_source.original').read_bytes());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen3VLModel')
init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
attrs={t.attr for n in init.body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self'}
assert {'visual','language_model'}<=attrs and spec['frozen_paths']==['model.visual']
commands=[['-m','unittest','vlm.tests.test_native_recipe_contract','vlm.tests.test_qwen_native_recipe_contract','vlm.tests.test_diagnostic_outputs','vlm.tests.test_hygon_vlm_recipe'],['-m','unittest','discover','-s','tests'],['tools/check_repository.py','--worktree']]
checks=[]
for ordinal,args in enumerate(commands):
 start=time.monotonic();r=subprocess.run([sys.executable,'-B','-X','utf8',*args],cwd=F,capture_output=True)
 (D/('local-check-%d.stdout.txt'%ordinal)).write_bytes(r.stdout);(D/('local-check-%d.stderr.txt'%ordinal)).write_bytes(r.stderr)
 checks.append(dict(args=args,returncode=r.returncode,wall_seconds=time.monotonic()-start,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest()))
 assert r.returncode==0
result=dict(passed=True,original_launch_SHA256=hashlib.sha256(raw).hexdigest(),spec=spec,checks=checks,
 observed_source_metadata=observed,actual_Torch=False,actual_Swift_PEFT=False,weight_values_read=False,GPU_calls=0,
 logit_F32_bytes_at_full4096=4096*config['text_config']['vocab_size']*4,
 full4096_logit_fits_current_per_artifact_cap=4096*config['text_config']['vocab_size']*4<=2*1024**3-1,
 actual_first_batch_logit_bytes=None,CPU_target_GO=False,GPU_GO=False,Model_GO=False)
with (D/'local-check-result.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
print(json.dumps(dict(passed=True,check_groups=len(checks),original_index_tensors=750,weight_values_read=False,GPU_calls=0,full4096_F32_logit_exceeds_cap=True)))
