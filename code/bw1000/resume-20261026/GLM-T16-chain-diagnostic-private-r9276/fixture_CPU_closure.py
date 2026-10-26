import pathlib,json,os,ast,importlib,subprocess,hashlib
p=pathlib.Path(__file__).parent;import torch;assert not torch.cuda.is_initialized();m=importlib.import_module('GPU_run');assert pathlib.Path(m.__file__).parent==p and not torch.cuda.is_initialized();proof=json.loads(pathlib.Path('/refactor/candidates/GLM-irregular-vector-r9270b/CPU-build.json').read_bytes());assert proof['completed'];z=subprocess.run(['ldd',proof['library']],text=True,capture_output=True,timeout=30);assert z.returncode==0 and 'not found'not in z.stdout+z.stderr
for f in p.glob('*.py'):ast.parse(f.read_text())
for f in p.glob('*.sh'):subprocess.run(['bash','-n',str(f)],check=True)
index=json.loads(pathlib.Path('/old/models/ZhipuAI-GLM-5.3/model.safetensors.index.json').read_bytes())['weight_map'];keys=['model.layers.38.mlp.gate.weight','model.layers.38.mlp.gate.e_score_correction_bias']+[f'model.layers.38.mlp.experts.{e}.{proj}.{suffix}'for e in range(256)for proj in ['gate_proj','up_proj','down_proj']for suffix in ['weight','weight_scale_inv']];assert all(k in index for k in keys)
from trace_summary_r9276 import summarize,union_us
assert union_us([(0,3),(2,4),(6,8)])==6
fake={'traceEvents':[{'ph':'X','name':'GLM_CHAIN_DIAGNOSTIC','ts':0,'dur':20},{'ph':'X','name':'K','cat':'kernel','ts':2,'dur':5,'args':{'device':0}},{'ph':'X','name':'L','cat':'kernel','ts':5,'dur':4,'args':{'device':0}}]};assert summarize(fake)['kernel_interval_union_seconds']==7e-6
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])as profile:
 with torch.profiler.record_function('CPU_import_actual'):torch.ones(3).sum()
assert len(profile.events())>0 and torch.profiler.kineto_available()
q={'passed':True,'kineto_available':torch.profiler.kineto_available(), 'supported_activities':str(torch.profiler.supported_activities()),'CPU_only':True,'GPU_initialized':torch.cuda.is_initialized(),'dlopen':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'node':os.uname().nodename,'entry_SHA256':hashlib.sha256((p/'GPU_run.py').read_bytes()).hexdigest(),'checkpoint_keys':len(keys),'native_library_SHA256':proof['library_SHA256'],'ldd':z.stdout};(p/'fixture-CPU-result.json').write_text(json.dumps(q,indent=2));print(json.dumps(q))
