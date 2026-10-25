import pathlib,json,subprocess,hashlib,shlex,os,traceback
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-KDA-only-serial-packed-r9025';i=r/'incoming';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[],'environment':{n:os.environ.get(n)for n in ['LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH','TORCH_LIB_DIR']}}
try:
 exe=r/'candidates/Kimi-KDA-only-serial-packed-r9020c/supported_metadata';q['same_exe_SHA256']=sha(exe)
 for name,cmd in [('readelf',['readelf','-d',str(exe)]),('ldd',['ldd',str(exe)])]:
  p=subprocess.run(cmd,text=True,capture_output=True,timeout=20);q[name]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr};assert p.returncode==0,(name,p.stderr)
 assert 'not found'not in q['ldd']['stdout'],q['ldd']
 p=subprocess.run([str(exe)],text=True,capture_output=True,timeout=20);q['metadata_original_and_packed']={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr};assert p.returncode==0,(p.stdout,p.stderr)
 q['metadata_result']=json.loads(p.stdout);assert not q['metadata_result']['original_narrow_supports']and q['metadata_result']['only_packed_out_supports']
 b.mkdir(exist_ok=False);src=b/'kda_conv_native.cpp';src.write_bytes((i/'kda_conv_native_r9020b.cpp').read_bytes());out=b/'libKimi_KDA_only_serial_packed_r9025.so'
 old=json.loads((r/'candidates/Kimi-KDA-only-serial-r9020/CPU-build.json').read_text());cmd=old['commands'][0][:];cmd[cmd.index('-MF')+1]=str(b/'native.d');cmd[cmd.index('-MF')+2]=str(src);cmd[cmd.index('-o')+1]=str(out);q['commands'].append(cmd)
 p=subprocess.run(cmd,text=True,capture_output=True,timeout=120);q['build']={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr};assert p.returncode==0,p.stderr
 symbols=subprocess.check_output(['nm','-D','--defined-only',str(out)],text=True);assert all(n in symbols for n in ['kda_conv_create','kda_conv_run','kda_conv_run_serial_kda','kda_conv_close','kda_conv_counts','kda_conv_error'])
 deps=shlex.split((b/'native.d').read_text().replace(chr(92)+chr(10),' ').split(':',1)[1]);q.update(completed=True,library=str(out),library_SHA256=sha(out),source_SHA256=sha(src),exports=symbols,dependencies_SHA256={n:sha(n)for n in deps if pathlib.Path(n).is_file()})
except BaseException as e:q['error']=repr(e);q['traceback']=traceback.format_exc()
p=r/'results'/('Kimi-KDA-packed-CPU-r9025-'+q['producer']+'.json');assert not p.exists();p.write_text(json.dumps(q,indent=2));print(json.dumps({k:v for k,v in q.items()if k not in ['commands','exports','dependencies_SHA256']}));assert q['completed']
