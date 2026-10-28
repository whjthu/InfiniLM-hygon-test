"""Bounded HOST compilation/link only; no Torch, DSO loading, weights or GPU."""
import pathlib,json,hashlib,subprocess,shlex,os,time,traceback
from CPU_static_r172 import check
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');B=R/'candidates/Deep-C128-weight-cache-r172'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
start=time.monotonic();deadline=start+900
assert not B.exists();B.mkdir()
q=dict(completed=False,CPU_only=True,GPU_initialized=False,dlopen=False,GPU_TUs_compiled=0,producer=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),commands=[])
def run(argv,label,limit=180):
 left=deadline-time.monotonic();assert left>0,'900s CPU deadline'
 q['commands'].append(argv);z=subprocess.run(argv,text=True,capture_output=True,timeout=min(left,limit));(B/(label+'.log')).write_text(z.stdout+z.stderr)
 assert z.returncode==0,(label,z.stderr[-4000:]);return z.stdout
def deps(s):return {str(pathlib.Path(p).resolve())for p in shlex.split(s.replace('\\\n',' ').split(':',1)[1])}
def prefix(a):
 out=[];i=0
 while i<len(a):
  x=a[i]
  if x in ['-o','-MF','-MT','-MQ']:i+=2;continue
  if x in ['-c','-MD','-MMD','-M','-MM']or x.endswith('.cpp'):i+=1;continue
  out.append(x);i+=1
 return out
try:
 q['source_contract']=check();m=json.loads((D/'manifest-r172.json').read_bytes());c=json.loads((D/'source-context-r172.json').read_bytes())
 receipt=R/'candidates/deep-c4-smallt-model-r9254/CPU-build.json';assert sha(receipt)==c['current_CPU_receipt_SHA256'];p=json.loads(receipt.read_bytes())
 assert p['completed'] and p['libraries']['full']['SHA256']==sha(p['libraries']['full']['path'])=='ac9370c8f188bfda8555718c52690b914883fc6b02b9267cbd279be3ecdd3835'
 old=c['current_link']['objects'];assert old==p['link']['full']['objects']and len(old)==48
 assert all(sha(n)==h for n,h in old.items())
 assert all(sha(n)==h for n,h in c['production_source_pins'].items())
 for n,h in p['actual_DTK_bindings'].items():assert sha(n)==h['SHA256']
 replacement_keys={u['object']for u in c['units']};assert len(replacement_keys)==9
 # Fail closed on an additional complete-type consumer missed by historical .d files.
 for i,obj in enumerate(old):
  syms=run(['nm','-C',obj],'abi-scan-'+str(i),20)
  assert 'TemporalCompressor'not in syms or obj in replacement_keys,('unrecompiled-TemporalCompressor-consumer',obj)
 private={x.name:str(x.resolve())for x in (D/'src').iterdir()if x.is_file()};compiled=[]
 for i,u in enumerate(c['units']):
  src=D/u['private_source'];assert old[u['object']]==u['object_SHA256'];assert sha(u['source'])==u['source_SHA256']
  flags=prefix(u['compile_argv']);flags=[flags[0],'-I'+str(D/'src'),'-I'+str(D),*flags[1:]]
  obj=B/(str(i)+'.o');md=B/(str(i)+'.d');ds=deps(run(flags+['-M',str(src)],str(i)+'-M'))
  assert str(src.resolve())in ds
  for n in ds:
   if pathlib.Path(n).name in private:assert n==private[pathlib.Path(n).name],('foreign-private-header',n)
  if pathlib.Path(src).name not in ['deepseek_v4_swa_attention.cpp','deepseek_v4_paged_latent_cache_public.cpp']:
   assert str((D/'src/deepseek_v4_temporal_compressor.hpp').resolve())in ds
  pin={n:sha(n)for n in ds};argv=flags+['-MD','-MF',str(md),'-c',str(src),'-o',str(obj)];run(argv,str(i)+'-compile')
  assert ds==deps(md.read_text())and all(sha(n)==h for n,h in pin.items())
  compiled.append(dict(source=str(src),source_SHA256=sha(src),replaces=u['object'],object=str(obj),object_SHA256=sha(obj),compile_argv=argv,dependency_SHA256=pin,fresh_M_equal_MD=True))
 replacement={u['replaces']:u['object']for u in compiled};link=[replacement.get(x,x)for x in c['current_link']['argv']];lib=B/'libDeep_C128_weight_cache_r172.so';link[link.index('-o')+1]=str(lib);run(link,'link',120)
 oldnm=run(['nm','-D','--defined-only',p['libraries']['full']['path']],'old-exports',30);newnm=run(['nm','-D','--defined-only',str(lib)],'new-exports',30)
 types=lambda s:{v.split()[-1]:v.split()[-2]for v in s.splitlines()if len(v.split())>=3}
 oldtypes,newtypes=types(oldnm),types(newnm);missing=set(oldtypes)-set(newtypes)
 known={'_ZNK3fmt3v1217basic_format_argsINS0_7contextEE3getEi','_ZZN3fmt3v126detail12format_floatIdEEiT_iRKNS0_12format_specsEbRNS1_6bufferIcEEENKUljPcE_clEjSA_'}
 assert missing<=known and all(oldtypes[n]=='W'for n in missing),missing
 required=['modern_deep_c128_weight_cache_mode_r172','modern_deep_c128_weight_cache_stats_r172'];assert set(required)<=set(newtypes)
 ld=run(['ldd',str(lib)],'ldd',30);assert 'not found'not in ld
 dyn=run(['readelf','-d',str(lib)],'readelf',30);assert 'SYMBOLIC'in dyn
 assert all(sha(n)==h for n,h in old.items())
 q.update(completed=True,library=str(lib),library_SHA256=sha(lib),compiled=compiled,link=link,HOST_TUs_compiled=len(compiled),reused_objects_SHA256={n:h for n,h in old.items()if n not in replacement},actual_DTK_bindings=p['actual_DTK_bindings'],required_exports=required,exact_private_fmt_weak_exceptions={n:oldtypes[n]for n in missing},ldd=ld,default_OFF=True,new_resident_bytes_per_owner=335544320,source_manifest_SHA256=sha(D/'manifest-r172.json'),scope='HOST source/ABI/dependency/build/link closure only. C1 numerical/performance/Model/GPU remain UNQUALIFIED. No component or Model launch.')
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
q['seconds']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps({k:q.get(k)for k in ['completed','producer','library_SHA256','error','seconds']}));assert q['completed']
