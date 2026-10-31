"""Same-container ldd comparison only: no project dlopen/import/weights/Model."""
from pathlib import Path
import json,hashlib,subprocess,os,re,datetime,importlib.util
R=Path('/refactor');D=R/'incoming/common-dso-audit-r301';C=R/'candidates/common-dso-audit-r301';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_CPU':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'CPU_only':True,'GPU':0,'Model':0,'HTTP':0,'weights':0,'Native_dlopen':0,'runtime_imports':0,'LD_PRELOAD_cleared_for_CPU':True,'phases':[]}
assert not os.environ.get('LD_PRELOAD')
base_env=dict(os.environ)
for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']:assert not base_env.get(k)
p=R/'candidates/latest-family-native-r269/CPU-build.json';assert sha(p)=='3399988b130529836f85826e912673173831e4dd06429073578949e3c6430fc5';build=json.loads(p.read_bytes())
for model,new,old,native,h in [('GLM',299,290,'candidates/GLM-PP4-registry-repair-r184b/libGLM_PP4_registry_r184b.so','038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'),('Deep',300,287,'candidates/latest-family-native-r269/Deep/libDeep_latest_native_r269.so','afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2')]:
 nd=R/'incoming'/(model+'-latest-full-owner-r'+str(new));od=R/'incoming'/(model+'-latest-full-owner-r'+str(old));name='GPU_inside_r290.sh'if model=='GLM'else'GPU_inside_r277.sh'
 sm=json.loads((nd/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(nd/n)==v for n,v in sm['files_SHA256'].items());assert sha(R/native)==h
 spec=importlib.util.spec_from_file_location('mapping_'+model,nd/'common_mapping_audit.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 pins=mod.expected(model);assert all(sha(p)==v for p,v in pins.items())
 e=build['families'][model]['Common_extension'];assert pins[e['new']]==e['new_SHA256']
 runtime=next(p for p in pins if p.endswith('libinfinilm_runtime.so'));assert pins[runtime]==e['runtime_library_SHA256']
 phases={}
 for network in (['Socket','shca']if model=='GLM'else['retained']):
  for label,d in [('control',od),('candidate',nd)]:
   script=(d/name).read_text();stop='python3 /refactor/incoming/GLM_transient_preflight_r9059.py'if model=='GLM'else'python -B /refactor/incoming/deep_affinity_env_contract_r7166.py';assert script.count(stop)==1;prefix=script[:script.index(stop)]
   prefix=re.sub(r'^export TORCH_LIB_DIR=.*$', 'export TORCH_LIB_DIR=/usr/local/lib/python3.10/dist-packages/torch/lib',prefix,flags=re.M)
   prefix=re.sub(r'^export LD_PRELOAD=.*$', 'unset LD_PRELOAD',prefix,flags=re.M)
   assert 'python -c'not in prefix and 'ctypes'not in prefix
   env={**base_env,'GLM_EXACT_BANKS_R4188':'0','GLM_CHANNELS_R4363':'8','GLM_BOUNDED_ROPE_R4279':'1','GLM_SHARED_CONTROL_R4279':'1','GLM_NETWORK_TRIAL':network}
   z=subprocess.run(['/bin/bash','--noprofile','--norc','-c',prefix+"\npython3 -B -c 'import os,json;print(json.dumps(dict(os.environ)))'"],env=env,text=True,capture_output=True,timeout=20);assert z.returncode==0,z.stderr[-1200:];resolved_env=json.loads(z.stdout.splitlines()[-1]);assert not resolved_env.get('LD_PRELOAD')
   for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']:assert not resolved_env.get(k)
   if model=='GLM'and label=='candidate':assert resolved_env['LD_LIBRARY_PATH'].split(':')[0]=='/refactor/candidates/latest-family-native-r269/GLM/python'
   records={}
   for lib in [str(R/native),e['new'],runtime]:
    z=subprocess.run(['ldd',lib],env=resolved_env,text=True,capture_output=True,timeout=20);assert z.returncode==0 and 'not found'not in z.stdout+z.stderr,(lib,z.stdout,z.stderr)
    resolutions={m[1]:m[2]for line in z.stdout.splitlines()if(m:=re.match(r'^\s*(\S+)\s+=>\s+(\S+)',line))}
    records[lib]={'rc':z.returncode,'resolved':resolutions,'stdout_SHA256':hashlib.sha256(z.stdout.encode()).hexdigest(),'stderr_SHA256':hashlib.sha256(z.stderr.encode()).hexdigest()}
   native_map=records[str(R/native)]['resolved'];extkey=Path(e['new']).name;assert native_map[extkey]==(e['original']if model=='GLM'and label=='control'else e['new'])
   assert native_map['libinfinilm_runtime.so']==runtime
   phases[(network,label)]=records
   q['phases'].append({'model':model,'network':network,'phase':label,'owner_manifest_SHA256':sha(d/'SOURCE_MANIFEST.json'),'LD_LIBRARY_PATH':resolved_env['LD_LIBRARY_PATH'],'actual_Native_SHA256':h,'libraries':records,'fixed_Common_runtime_SHA256':pins})
  a=phases[(network,'control')];b=phases[(network,'candidate')]
  for lib in a:
   diff={k for k in a[lib]['resolved'].keys()|b[lib]['resolved'].keys()if a[lib]['resolved'].get(k)!=b[lib]['resolved'].get(k)}
   assert diff==({Path(e['new']).name}if model=='GLM'and lib==str(R/native)else set()),(model,lib,diff)
q.update(completed=True,passed=True,Model_qualified=False,qualification='Actual same-container ldd resolution and pinned file bytes only; no live project symbol binding, GPU imports or model lifecycle tested')
p=C/'CPU-inside.json';assert not p.exists();p.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
