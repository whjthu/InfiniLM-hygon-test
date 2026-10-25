import pathlib,json,hashlib,shutil,runpy,contextlib,io,subprocess,time,shlex,os
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-attnres-front-r8982';inc=r/'incoming'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
d={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[]};started=time.monotonic()
try:
 m=json.loads((inc/'Kimi-attnres-front-source-r8983.json').read_text())
 b.mkdir(exist_ok=False);(b/'src').mkdir()
 for n,h in m['source_SHA256'].items():
  assert sha(inc/n)==h,n
  if n in ['attnres_front_r8982.cu','attnres_explicit_r8982.cpp']:shutil.copyfile(inc/n,b/'src'/n)
 out=io.StringIO()
 with contextlib.redirect_stdout(out):runpy.run_path(str(inc/'cpu_build_plan_r8982.py'))
 plan=json.loads(out.getvalue())
 for i,cmd in enumerate(plan['commands']):
  d['commands'].append(cmd)
  with(b/('build'+str(i)+'.log')).open('x')as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,240-(time.monotonic()-started)))
  assert z.returncode==0,(i,(b/('build'+str(i)+'.log')).read_text()[-9000:])
 native=b/'libattnres_front_r8982.so';defs=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True)
 for n in ['modern_attnres','modern_attnres_front_phase_r8982','modern_attnres_front_mode_r8982','modern_attnres_front_count_r8982','modern_attnres_shutdown_r8982','kimi_attnres_front_launch_r8982']:assert n in defs,n
 dep={}
 for md in b.glob('*.d'):
  files=shlex.split(md.read_text().replace('\\\n',' ').split(':',1)[1]);dep[md.name]={p:sha(p)for p in sorted(set(files))if pathlib.Path(p).is_file()}
 common=r/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM';sdk=r/'install/hygon-kimi-5f91ad7-r3672'
 libs=[common/'python/_infinilm.cpython-310-x86_64-linux-gnu.so',common/'python/infinilm/_lib/libinfinilm_runtime.so',*[sdk/'lib'/n for n in ['libinfiniops.so','libinfinirt.so','libinfiniccl.so']]]
 d.update(completed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],native=str(native),native_SHA256=sha(native),source_SHA256=m['source_SHA256'],fresh_MD=dep,actual_dependency_SHA256={str(p):sha(p)for p in libs},defined=defs,needed=subprocess.check_output(['readelf','-d',str(native)],text=True),build_wall_s=time.monotonic()-started)
except BaseException as e:d['error']=repr(e);raise
finally:
 d['elapsed_s']=time.monotonic()-started;p=r/'results/Kimi-attnres-front-CPU-r8983.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print(json.dumps({k:d.get(k)for k in ['completed','step','error','native_SHA256','elapsed_s']}))
