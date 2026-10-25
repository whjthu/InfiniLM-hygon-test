import pathlib,json,hashlib,subprocess,os,shlex
r=pathlib.Path('/refactor');src=pathlib.Path(__file__).resolve().parent;b=r/'candidates/Kimi-taps-fixture-r9023';b.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
parent=json.loads((r/'candidates/Kimi-target-taps-r9022/CPU-build.json').read_text());assert parent['completed'] and sha(parent['native'])==parent['native_SHA256']=='2caae7170e840a9cf51c0e32758c6b093c6a6160a8897aea15b8b1fa0826d3e1'
flags=parent['commands'][0][:parent['commands'][0].index('-MD')];native=b/'libtaps_fixture_r9023.so';deps=b/'fixture.d'
link=[x for x in parent['commands'][-1][1:] if not x.endswith('.o') and not x.startswith('-Wl,-Map,')];link[link.index('-o')+1]=str(native)
cmd=[*flags,'-MD','-MF',str(deps),str(src/'taps_fixture_r9023.cpp'),parent['native'],*link];q={'completed':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'CPU_only':True,'GPU':False,'dlopen':False,'commands':[cmd]}
try:
 p=subprocess.run(cmd,text=True,capture_output=True,timeout=150);(b/'build.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr[-6000:]
 symbols=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True);assert all(n in symbols for n in ['taps_fixture_create_r9023','taps_fixture_call_r9023','taps_fixture_close_r9023'])
 names=shlex.split(deps.read_text().replace('\\\n',' ').split(':',1)[1]);q.update(completed=True,native=str(native),native_SHA256=sha(native),parent_native=parent['native'],parent_native_SHA256=parent['native_SHA256'],source_SHA256=sha(src/'taps_fixture_r9023.cpp'),dependencies_SHA256={n:sha(n)for n in names if pathlib.Path(n).is_file()})
except BaseException as e:q['error']=repr(e)
(b/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps(q));assert q['completed']
