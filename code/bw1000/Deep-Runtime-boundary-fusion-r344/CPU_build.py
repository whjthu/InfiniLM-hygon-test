import pathlib,json,subprocess,hashlib,shlex,time,os,traceback
S=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');B=R/'candidates/Deep-Runtime-boundary-fusion-r344';assert not B.exists();B.mkdir();start=time.monotonic()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q=dict(completed=False,CPU_only=True,GPU_initialized=False,producer=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
def run(a,label):
 z=subprocess.run(a,text=True,capture_output=True,timeout=180);(B/(label+'.log')).write_bytes((z.stdout+z.stderr).encode());assert z.returncode==0,(label,z.stderr[-5000:]);return z.stdout
def deps(p):return set(shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1]))
try:
 c=json.loads((S/'build-context.json').read_text());assert sha(c['parent_receipt'])==c['parent_receipt_SHA256'];old=json.loads(pathlib.Path(c['parent_receipt']).read_text());objects=old['link']['full']['objects'];assert all(sha(p)==h for p,h in objects.items());assert sha(c['kernel'])==c['kernel_SHA256'];u=c['unit'];assert sha(u['source'])==c['parent_compressor_SHA256'];assert u['object']in objects
 flags=u['compile_argv'][:u['compile_argv'].index('-MD')];compiled=[]
 for n in ['compressor344.cpp','bridge344.cpp']:
  src=S/n;obj=B/(n+'.o');mf=B/(n+'.M');md=B/(n+'.d');run(flags+['-M','-MF',str(mf),'-MT',str(obj),str(src)],n+'-deps');pin={p:sha(p)for p in deps(mf)};run(flags+['-MD','-MF',str(md),'-c',str(src),'-o',str(obj)],n+'-compile');assert deps(md)==set(pin)and all(sha(p)==h for p,h in pin.items());compiled.append(dict(source=str(src),SHA256=sha(src),object=str(obj),object_SHA256=sha(obj),dependencies=pin))
 lib=B/'libDeep344.so';argv=[compiled[0]['object']if x==u['object']else x for x in old['link']['full']['argv']];argv[argv.index('-o')+1]=str(lib);argv[argv.index('-o'):argv.index('-o')]=[compiled[1]['object'],c['kernel'],'-Wl,-rpath,'+str(pathlib.Path(c['kernel']).parent)];run(argv,'link');ld=run(['ldd',str(lib)],'ldd');assert'not found'not in ld;nm=run(['nm','-D','--defined-only',str(lib)],'exports');assert all(n in nm for n in ['modern_deep344_mode','modern_deep344_calls','modern_real_compress_create','modern_real_compress_forward','modern_real_compress_snapshot']);assert all(sha(p)==h for p,h in objects.items());q.update(completed=True,library=str(lib),library_SHA256=sha(lib),compiled=compiled,link=argv,parent_library=old['libraries']['full'],kernel_SHA256=c['kernel_SHA256'],ldd=ld,ABI='unchanged class layout; source-local atomics only; original full48 objects except compressor replaced; bridge added',reused_objects={p:h for p,h in objects.items()if p!=u['object']})
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
q['CPU_wall_s']=time.monotonic()-start;(B/'CPU-build.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({k:q.get(k)for k in ['completed','producer','CPU_wall_s','library_SHA256','error']}));assert q['completed']
