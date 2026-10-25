import pathlib,subprocess,json,os,hashlib
p=pathlib.Path('/opt/dtk/bin/hipprof');q={'CPU_only':True,'no_GPU_init':True,'path':str(p),'exists':p.exists(),'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
if p.exists():
 z=subprocess.run([str(p),'--help'],capture_output=True,text=True,timeout=25);q.update(help_returncode=z.returncode,help=z.stdout+z.stderr,SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
 q['libs']=[str(x)for x in pathlib.Path('/opt/dtk').glob('lib*/**/*')if x.is_file()and any(v in x.name for v in ['roctracer','hipprof','hip_trace'])][:40]
q['supports_flags']=all(x in q.get('help','')for x in ['--hip-trace','--trace-off','--output-type'])
p=pathlib.Path('/refactor/results/Deep-trace-CPU-r9087.json');assert not p.exists();p.write_text(json.dumps(q,indent=2));print(json.dumps(q))
