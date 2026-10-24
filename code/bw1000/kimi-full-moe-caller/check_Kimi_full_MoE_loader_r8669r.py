import pathlib,subprocess,json,hashlib,os
r=pathlib.Path('/refactor');lib=r/'candidates/Kimi-full-MoE-caller-r8669p/libKimi_full_MoE_caller_r8669p.so'
z=subprocess.run(['ldd',str(lib)],capture_output=True,text=True,timeout=30)
d={'scope':'CPU_ELF_dependency_resolution_only_no_dlopen_no_GPU_init','step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'library_SHA256':hashlib.sha256(lib.read_bytes()).hexdigest(),'returncode':z.returncode,'ldd':z.stdout,'stderr':z.stderr,'passed':z.returncode==0 and 'not found'not in z.stdout,'helper_SHA256':hashlib.sha256((r/'candidates/kimi-helper-bounded-bf16-r6054/libkimi_grouped_prefill.so').read_bytes()).hexdigest()}
p=r/'results/Kimi-full-MoE-loader-check-r8669r.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print(json.dumps(d));assert d['passed']
