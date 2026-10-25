import pathlib,json,subprocess,hashlib,os,shlex,shutil
r=pathlib.Path('/refactor');i=r/'incoming/Kimi-modes-r9007';b=r/'candidates/Kimi-ordinary-pair-r9007';b.mkdir();src=b/'src';src.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'native_host_TUs':1,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[]}
def run(args,label):
 q['commands'].append(args);p=subprocess.run(args,capture_output=True,text=True,timeout=130);(b/(label+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,(label,p.stderr[-7000:]);return p.stdout
try:
 for n in ['kimi_pair_probe_r9006.cpp','kimi_pair_state_r9006.hpp','check_pair_state_r9006.cpp']:shutil.copyfile(i/n,src/n)
 original=r/'candidates/kimi-compact-cache-r4974/kimi_compact_probe_r4974.cpp';assert sha(original)=='0371f6e9bfbeddb4fabe8322b7dc904a9830129021d8a74c0d0ac13b5624eb21'
 freeze=r/'candidates/kimi-compact-cache-r4974/kimi_engine_input_r145.hpp';assert sha(freeze)=='9ec9ef4c3c2ff5af60474729f156c16f24c6a2fb16c15510585eb08ee5d278a4';shutil.copyfile(freeze,src/freeze.name)
 parent=json.loads((r/'results/Kimi-Model-events-CPU-r8669zq1.json').read_text());assert parent['completed'];assert sha(parent['native'])==parent['native_SHA256']=='2c2961d29d56788c4e59061b241dc108bc0e9bff3153ed8d7aa6f3f575b83444'
 for n,h in parent['all_linked_object_SHA256'].items():assert sha(n)==h,n
 test=b/'test_state';run(['g++','-std=c++17','-O2','-I'+str(src),str(src/'check_pair_state_r9006.cpp'),'-o',str(test)],'CPU_test_compile');q['CPU_contract']=json.loads(run([str(test)],'CPU_test_run'))
 old=json.loads((r/'results/Kimi-I192-smoke-CPU-r8669zf.json').read_text());flags=old['commands'][0][:old['commands'][0].index('-MD')];flags=[flags[0],'-I'+str(src),*flags[1:]]
 obj=b/'pair_probe.o';run([*flags,'-MD','-MF',str(b/'pair_probe.d'),'-c',str(src/'kimi_pair_probe_r9006.cpp'),'-o',str(obj)],'native_compile')
 link=list(parent['commands'][-1]);replaced='/refactor/candidates/kimi-compact-cache-r4974/objects/kimi_engine_probe_r2475.o';assert link.count(replaced)==1;link[link.index(replaced)]=str(obj);native=b/'libKimi_pair_r9007.so';link[link.index('-o')+1]=str(native);link=[('-Wl,-Map,'+str(b/'link.map'))if a.startswith('-Wl,-Map,')else a for a in link];run(link,'native_link')
 nm=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True);assert all(n in nm for n in ['modern_kimi_engine_create_pair_r9006','modern_kimi_pair_geometry_r9006','modern_kimi_pair_rows_r9006','modern_kimi_pair_prefill_r9006','modern_kimi_pair_forward_r9006','modern_kimi_pair_commit_r9006','modern_kimi_pair_reset_r9006','modern_kimi_pair_mode_r9007','modern_kimi_pair_ordinary_forward_r9007'])
 unchanged={n:h for n,h in parent['all_linked_object_SHA256'].items() if n!=replaced};assert all(sha(n)==h for n,h in unchanged.items());deps=shlex.split((b/'pair_probe.d').read_text().replace('\\\n',' ').split(':',1)[1]);q.update(completed=True,native=str(native),native_SHA256=sha(native),source_SHA256={p.name:sha(p)for p in src.iterdir()},original_CAPI_SHA256=sha(original),original_freeze_SHA256=sha(freeze),replaced_object=replaced,unchanged_object_SHA256=unchanged,dependencies_SHA256={n:sha(n)for n in deps if pathlib.Path(n).is_file()},exports='\n'.join(a for a in nm.splitlines()if 'r9006'in a or 'r9007'in a),readelf=subprocess.check_output(['readelf','-d',str(native)],text=True))
except BaseException as e:q['error']=repr(e)
(b/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps({k:v for k,v in q.items()if k not in ['commands','dependencies_SHA256','unchanged_object_SHA256','readelf']}));assert q['completed']
