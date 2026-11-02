"""HOST-only thin bridge to sole a314 Native; no dlopen, HIP rebuild or model."""
import pathlib,json,hashlib,subprocess,os,time,shlex,re,traceback
R=pathlib.Path('/refactor');S=pathlib.Path(__file__).resolve().parent;B=R/'candidates/Deep-HC-Native-dispatch-qual-r378-cpu-v1';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q=dict(completed=False,passed=False,CPU_only=True,GPU_initialized=False,Native_dlopen=False,Runtime_created=False,weights_payload_opened=False,Model_loaded=False,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],commands=[],publication_commit=os.environ['DEEP378_PUBLICATION_COMMIT']);started=time.perf_counter()
def run(a,label,timeout=90):
 q['commands'].append(a);z=subprocess.run(a,capture_output=True,text=True,timeout=timeout);(B/(label+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(label,z.returncode,z.stderr[-3000:]);return z.stdout
try:
 assert re.fullmatch('[0-9a-f]{40}',q['publication_commit'])
 mp=S/'SOURCE_MANIFEST.json';assert sha(mp)==os.environ['DEEP378_CPU_MANIFEST_SHA'];m=json.loads(mp.read_bytes());assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())
 rp=R/'candidates/Deep-HC-fullpre-Native-r377-cpu-v1/CPU-build.json';assert sha(rp)=='837b0e824e79a33a460754eedc943a8c11786acfa856dde5ca2c438351ab5389';native_receipt=json.loads(rp.read_bytes());native=pathlib.Path(native_receipt['library']);assert native_receipt['passed']and sha(native)=='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'
 for p,h in native_receipt['all50_objects_SHA256'].items():assert sha(p)==h
 bp=R/'candidates/Deep-HC-epilogue-r358-cpu-r2/CPU-build.json';assert sha(bp)=='0a963eafd1941104823a2567b9dfbffb900372be6a6479779104780a372170b1';base=json.loads(bp.read_bytes());assert base['completed']and not base['Native_dlopen']and not base['GPU_initialized']
 old=R/'incoming/Deep-HC-epilogue-r358/repair-r2';src=S/'bridge.cpp';obj=B/'bridge.o';md=B/'bridge.d';original=next(a for a in base['commands']if '-c'in a and any(str(v).endswith('/adapter.cpp')for v in a));cmd=list(original)
 for i,a in enumerate(cmd):
  if a=='-I'+str(old):cmd[i]='-I'+str(S)
  elif a.endswith('/adapter.cpp'):cmd[i]=str(src)
  elif a.endswith('/adapter.o'):cmd[i]=str(obj)
  elif a.endswith('/adapter.d'):cmd[i]=str(md)
 flags=[a for i,a in enumerate(cmd)if i not in [0,cmd.index('-c'),cmd.index('-c')+1,cmd.index('-o'),cmd.index('-o')+1,cmd.index('-MF'),cmd.index('-MF')+1]and a!='-MD']
 parse=lambda t:set(shlex.split(t.replace('\\\n',' ').split(':',1)[1]));fresh=parse(run(['g++',*flags,'-M',str(src)],'fresh-M'));deps={p:sha(p)for p in fresh}
 for p,h in deps.items():
  if not pathlib.Path(p).is_relative_to(S):assert base['dependencies_SHA256'].get(p)==h,('Originaldependencychanged',p)
 hc_headers=[p for p in deps if p.endswith('/deepseek_v4_hyper_connection.hpp')];assert len(hc_headers)==1 and native_receipt['dependencies_SHA256'][hc_headers[0]]==deps[hc_headers[0]]
 run(cmd,'bridge-compile');assert parse(md.read_text())==fresh and all(sha(p)==h for p,h in deps.items())
 oldlink=next(a for a in base['commands']if '-shared'in a);libraries=[a for a in oldlink if a.startswith(('-L','-l','-Wl,'))and not a.startswith('-Wl,-rpath,')and a!='-Wl,-Bsymbolic'];lib=B/'libDeep_HC_Native_bridge_r378.so'
 link=['g++',str(obj),str(native),'-shared','-fPIC',*libraries,'-Wl,--no-undefined','-Wl,-rpath,'+str(native.parent),'-o',str(lib)];run(link,'bridge-link')
 dyn=run(['readelf','-d',str(lib)],'dynamic');needed=[l.split('[',1)[1].split(']',1)[0]for l in dyn.splitlines()if'NEEDED'in l];assert str(native)in needed
 assert not any('libDeep'in n and n!=str(native)for n in needed)
 ld=run(['ldd',str(lib)],'ldd');assert 'not found'not in ld and str(native)in ld
 defined=run(['nm','-D','-C','--defined-only',str(lib)],'defined');undefined=run(['nm','-D','-C','--undefined-only',str(lib)],'undefined');assert 'HyperConnection::pre('not in defined and 'HyperConnection::pre('in undefined
 exports=['deep_hc_bridge_'+v+'_r378'for v in ['create','forward','state','destroy','error']];assert all(n in defined for n in exports)
 nsyms=run(['nm','-D','--defined-only',str(native)],'Native-exports');assert all(n in nsyms for n in ['deep_hc_fullpre_enable_r377','deep_hc_fullpre_enabled_r377','deep_hc_fullpre_clear_r377','deep_hc_fullpre_stats_r377','deep_hc_model_epilogue_set_r366','deep_hc_model_epilogue_counters_r366'])
 for p,v in native_receipt['actual_DTK_bindings'].items():assert sha(p)==v['SHA256']
 hip=native_receipt['existing_HIP_provider'];assert sha(hip['path'])==hip['SHA256']
 q.update(completed=True,passed=True,source_manifest_SHA256=sha(mp),source_SHA256=m['files_SHA256'],Native_receipt_SHA256=sha(rp),Native_path=str(native),Native_SHA256=sha(native),Native50_objects_SHA256=native_receipt['all50_objects_SHA256'],bridge_library=str(lib),bridge_SHA256=sha(lib),bridge_object_SHA256=sha(obj),dependencies_SHA256=deps,MD_equals_fresh_M=True,direct_NEEDED=needed,sole_Native_provider=True,HC_pre_undefined_in_bridge_resolved_by_a314=True,ldd=ld,exports=exports,actual_DTK_bindings=native_receipt['actual_DTK_bindings'],existing_HIP_provider=hip,link_argv=link,compile_inheritance_CPU700_receipt_SHA256=sha(bp),new_HIP_compilation=False,new_Native_compilation=False,GPU_qualification='NOT_RUN',Model_gain='UNMEASURED')
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
q['seconds']=time.perf_counter()-started;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_step','bridge_SHA256','error','seconds']}));assert q['passed']
