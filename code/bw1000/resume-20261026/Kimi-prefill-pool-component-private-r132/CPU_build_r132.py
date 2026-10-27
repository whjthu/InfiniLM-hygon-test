"""CPU-only actual two HOST builds: caller + fixture adapter. Zero GPU math compilation/dlopen."""
import pathlib,json,hashlib,subprocess,os,time,traceback,shlex
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;B=R/'candidates/Kimi-prefill-pool-component-r132';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();assert os.environ.get('KIMI_POOL_CPU_BUILD_R132')=='1';assert not B.exists();B.mkdir();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'new_GPU_TUs':0,'new_HOST_TUs':2,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[]};tick=time.monotonic()
def run(c,label,budget=180,env=None):
 q['commands'].append(c);z=subprocess.run(c,text=True,capture_output=True,timeout=budget,env=env);(B/(label+'.log')).write_bytes((z.stdout+z.stderr).encode());assert z.returncode==0,(label,z.stderr[-10000:]);return z
def semantics(c):
 out=[c[0]];i=1
 while i<len(c):
  x=c[i]
  if x in ['-o','-MF','-MT','-MQ','-MJ','-L','-l']:i+=2;continue
  if x in ['-c','-MD','-MMD','-MP','-MG','-M','-MM','-shared']:i+=1;continue
  if x.startswith(('-MF','-MT','-MQ','-MJ','-Wl,','-L','-l'))or x.endswith(('.cpp','.cc','.cxx','.o','.so','.a')):i+=1;continue
  out.append(x);i+=1
 return out
try:
 pins=json.loads((D/'actual-source-object-pins-r132.json').read_bytes());ctx=json.loads((D/'per-TU-context-r132.json').read_bytes());sourcepins=json.loads((D/'CPU-stage-source-manifest-r132.json').read_bytes())
 for n,h in sourcepins['files_SHA256'].items():assert sha(D/n)==h,n
 parent=json.loads(pathlib.Path(pins['parent_receipt']).read_bytes());assert parent['completed']and sha(pins['parent_receipt'])==pins['parent_receipt_SHA256']and sha(parent['native'])==parent['native_SHA256']==pins['parent_native_SHA256'];assert parent['all_linked_object_SHA256']==pins['all_linked_object_SHA256']
 for p,h in pins['all_linked_object_SHA256'].items():assert sha(p)==h,p
 for p,h in [(pins['helper'],pins['helper_SHA256']),(pins['helper_source'],pins['helper_source_SHA256']),(pins['Runtime'],pins['Runtime_SHA256']),(pins['RTlib'],pins['RTlib_SHA256']),(pins['Runtime_source'],pins['Runtime_source_SHA256']),(pins['pool_source'],pins['pool_source_SHA256'])]:assert sha(p)==h,p
 assert 'torch'not in __import__('sys').modules
 import importlib.util;spec=importlib.util.find_spec('torch');assert spec and spec.origin;torchlib=str(pathlib.Path(spec.origin).parent/'lib');assert pathlib.Path(torchlib).is_dir()
 for k in ['moe','component']:
  receipt=json.loads(pathlib.Path(ctx[k]['receipt']).read_bytes());assert receipt['completed']and ctx[k]['argv']in receipt['commands'],('perTU actualargv mismatch',k)
 cmd=ctx['moe']['argv'][:];assert all(x in cmd for x in ['-O2','-fPIC','-pthread']);cmd[cmd.index('-c')+1]=str(D/'src/kimi_moe_prefill_pool_trim_r130.cpp');cmd[cmd.index('-o')+1]=str(B/'moe.o');cmd[cmd.index('-MF')+1]=str(B/'moe.d');run(cmd,'caller')
 oldmoe=ctx['moe']['object'];assert pins['all_linked_object_SHA256'][oldmoe]==ctx['moe']['object_SHA256'];new=B/'libKimi_prefill_pool_native_r132.so';link=next(c[:]for c in parent['commands']if '-o'in c and c[c.index('-o')+1]==parent['native']);assert link.count(oldmoe)==1;link[link.index(oldmoe)]=str(B/'moe.o');link[link.index('-o')+1]=str(new)
 for i,x in enumerate(link):
  if x.startswith('-Wl,-Map'):link[i]='-Wl,-Map,'+str(B/'link.map')
 run(link,'native-link');objects={p:sha(p)for p in link if p.endswith('.o')};unchanged={p:h for p,h in pins['all_linked_object_SHA256'].items()if p!=oldmoe};assert all(objects.get(p)==h for p,h in unchanged.items())
 wrapper=B/'libKimi_prefill_pool_component_r132.so';wc=ctx['component']['argv'][:];oldsrc=next(x for x in wc if x.endswith('/W2_pair_runner_r9280.cpp'));wc[wc.index(oldsrc)]=str(D/'src/prefill_pool_component_r132.cpp');assert wc.count(ctx['component']['old_native'])==1;wc[wc.index(ctx['component']['old_native'])]=str(new);wc[wc.index('-o')+1]=str(wrapper);wc[wc.index('-MF')+1]=str(B/'component.d');wc=[('-MD'if x=='-MMD'else x)for x in wc];run(wc,'component')
 deps={};commands={'moe':cmd,'component':wc}
 for name,c in commands.items():
  paths=shlex.split((B/(name+'.d')).read_text().replace('\\\n',' ').split(':',1)[1]);deps[name]={p:sha(p)for p in paths if pathlib.Path(p).is_file()};fresh=B/(name+'.freshM.d');source=str(D/('src/kimi_moe_prefill_pool_trim_r130.cpp'if name=='moe'else'src/prefill_pool_component_r132.cpp'));run([*semantics(c),'-M','-MF',str(fresh),'-MT',str(B/(name+'.o')),source],name+'-freshM');fp=shlex.split(fresh.read_text().replace('\\\n',' ').split(':',1)[1]);assert set(fp)==set(paths),(name,'effective fullargv fresh-M mismatch')
 critical='/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/models/kimi_k3/kimi_k3_moe.hpp';assert deps['moe'][critical]==deps['component'][critical]
 defs=subprocess.check_output(['nm','-D','--defined-only',str(new)],text=True);olddefs=subprocess.check_output(['nm','-D','--defined-only',parent['native']],text=True);symbols={x.split()[-1]:x.split()[-2]for x in defs.splitlines()if len(x.split())>=3};assert symbols['modern_kimi_prefill_pool_trim_stats_r130']=='T';assert all(x.split()[-1]in symbols for x in olddefs.splitlines()if len(x.split())>=3)
 moe_symbols=subprocess.check_output(['nm','-C','--defined-only',str(B/'moe.o')],text=True);assert any(' T 'in x and 'KimiK3MoE::forward('in x for x in moe_symbols.splitlines())
 env=dict(os.environ,LD_LIBRARY_PATH=str(pathlib.Path(pins['helper']).parent)+':'+parent['LD_LIBRARY_PATH']+':'+torchlib);ldd=run(['ldd','-r',str(wrapper)],'ldd-r',60,env);text=ldd.stdout+ldd.stderr;assert 'not found'not in text and 'undefined symbol:'not in text and pins['helper']in text
 dt=subprocess.check_output(['readelf','-d',str(wrapper)],text=True);assert str(new)in dt and pins['parent_native']not in dt and ctx['component']['old_native']not in dt
 q.update(completed=True,native=str(new),native_SHA256=sha(new),library=str(wrapper),library_SHA256=sha(wrapper),all_linked_object_SHA256=objects,unchanged_object_SHA256=unchanged,actual_MD=deps,actual_perTU_argv=commands,critical_MoE_header_SHA256=deps['moe'][critical],helper_expected_path=pins['helper'],helper_expected_SHA256=pins['helper_SHA256'],ldd_relocations=text,wrapper_DT_NEEDED=dt,LD_LIBRARY_PATH=env['LD_LIBRARY_PATH'],Torch_find_spec_libpath=torchlib,CPU_Torch_imported=False,elapsed_s=time.monotonic()-tick,defaultOFF=True,Model_fit='PARK/UNKNOWN')
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-tick)
(B/'CPU-build.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({k:q.get(k)for k in ['completed','actual_step','native_SHA256','library_SHA256','error','elapsed_s']}));assert q['completed']
