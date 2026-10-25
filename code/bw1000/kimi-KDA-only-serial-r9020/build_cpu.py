import pathlib,json,hashlib,subprocess,os,shlex,time
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-KDA-only-serial-r9020';b.mkdir(exist_ok=False)
i=r/'incoming';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[]}
try:
 pin=json.loads((i/'Kimi-KDA-current-remote-r9004.json').read_text())
 for n,h in pin['files'].items():assert sha(n)==h,(n,'dependency changed')
 assert sha(pin['native'])==pin['native_sha']
 src=b/'kda_conv_native.cpp';src.write_bytes((i/'kda_conv_native_r9020.cpp').read_bytes())
 flags=pin['flags'][:pin['flags'].index('-MD')];out=b/'libKimi_KDA_only_serial_r9020.so'
 common=r/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM';sdk=r/'install/hygon-kimi-5f91ad7-r3672'
 cmd=[*flags,'-shared','-Wl,-Bsymbolic-functions','-MD','-MF',str(b/'native.d'),str(src),'-L'+str(common/'python'),'-l:_infinilm.cpython-310-x86_64-linux-gnu.so','-L'+str(common/'python/infinilm/_lib'),'-linfinilm_runtime','-L'+str(sdk/'lib'),'-linfiniops','-linfinirt','-L/opt/dtk/cuda/cuda/lib64','-lcudart','-Wl,--no-undefined','-o',str(out)]
 q['commands'].append(cmd);p=subprocess.run(cmd,capture_output=True,text=True,timeout=150);(b/'build.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr[-8000:]
 symbols=subprocess.check_output(['nm','-D','--defined-only',str(out)],text=True)
 assert all(n in symbols for n in ['kda_conv_create','kda_conv_run','kda_conv_close','kda_conv_counts','kda_conv_error','kda_conv_run_serial_kda'])
 deps=shlex.split((b/'native.d').read_text().replace('\\\n',' ').split(':',1)[1]);dep_sha={n:sha(n) for n in deps if pathlib.Path(n).is_file()}
 q.update(completed=True,library=str(out),library_SHA256=sha(out),source_SHA256=sha(src),dependencies_SHA256=dep_sha,exports=symbols,readelf=subprocess.check_output(['readelf','-d',str(out)],text=True),original_Model_DSO=pin['native'],original_Model_SHA256=pin['native_sha'],math='original implementation0 H6/D128 BF16 conv/Silu chunk; explicit KDA-only T1 selector; no new GPU kernel',linked_objects=[],scope='one new host TU; original dynamically linked Ops and runtime; no GPU initialization or dlopen')
except BaseException as e:q['error']=repr(e)
(b/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps({k:v for k,v in q.items() if k not in ['commands','dependencies_SHA256','exports','readelf']}));assert q['completed']
