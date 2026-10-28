"""Capacity prototype: private cache HOST + MoE HOST only, no GPU/Model entry."""
from pathlib import Path
import ast,json,hashlib,os,subprocess,time,shlex,traceback,sys
R=Path('/refactor');D=Path(__file__).resolve().parent;B=R/'candidates/Kimi-capacity-exact-cache-lease-r170';B.mkdir(exist_ok=False);start=time.monotonic();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q={'completed':False,'CPU_only':True,'GPU':False,'Native_loaded':False,'actual_step':os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),'commands':[],'Model_GO':False,'GPU_launch_allowed':False}
def run(cmd,label,seconds,env=None):
 q['commands'].append(cmd);p=subprocess.run(cmd,capture_output=True,text=True,timeout=seconds,env=env);(B/(label+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,(label,p.stderr[-2500:]);return p.stdout+p.stderr
try:
 m=json.loads((D/'source-manifest-r170.json').read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 for f in D.glob('*.py'):ast.parse(f.read_bytes())
 pins=json.loads((D/'parent-pins-r170.json').read_bytes());bp=Path(pins['CPU522_path']);assert sha(bp)==pins['CPU522_SHA256'];p=json.loads(bp.read_bytes());assert p['completed']and p['CPU_only']and not p['GPU']and not p['Native_loaded']and p['actual_step']=='910407.522'and sha(p['native'])==p['native_SHA256']==pins['Native522_SHA256']
 assert sha(R/'incoming/Kimi-TP32-I96-exact-storage-r167/kimi_moe_I96_exact_r167.cpp')==pins['source167_MoE_SHA256']
 for k in ['cache_base','cache_header','Runtime_header','provider_I96']:assert sha(pins[k+'_path'])==pins[k+'_SHA256'],k
 for path,h in pins['all_parent_objects_SHA256'].items():assert sha(path)==h,path
 for path,h in pins['parent_MD'].items():assert sha(path)==h,path
 q['CPU_source_lifetime_contract']=json.loads(run([sys.executable,'-B',str(D/'CPU_contract_r170.py'),pins['source_MoE163_dir']],'contract',20))
 cc=[x for x in p['commands']if '-c'in x and x[x.index('-c')+1].endswith('/kimi_moe_I96_exact_r167.cpp')];assert len(cc)==1;cc=cc[0]
 deps={}
 for name,source in [('moe','kimi_moe_capacity_r170.cpp'),('cache','kimi_cache_exact_r170.cpp')]:
  cmd=cc[:];cmd.insert(1,'-I'+str(D));cmd.insert(2,'-I'+str(Path(pins['cache_header_path']).parent));cmd[cmd.index('-c')+1]=str(D/source);cmd[cmd.index('-o')+1]=str(B/(name+'.o'));cmd[cmd.index('-MF')+1]=str(B/(name+'.d'));run(cmd,name+'_HOST',210)
  paths=shlex.split((B/(name+'.d')).read_text().replace(chr(92)+chr(10),' ').split(':',1)[1]);deps[name]={x:sha(x)for x in paths if Path(x).is_file()}
  assert deps[name].get(pins['Runtime_header_path'])==pins['Runtime_header_SHA256']or any(h==pins['Runtime_header_SHA256']for path,h in deps[name].items()if path.endswith('/runtime/runtime.hpp'))
 link=[x for x in p['commands']if '-shared'in x and '-o'in x and x[x.index('-o')+1]==p['native']];assert len(link)==1;link=link[0][:]
 oldmoe=next(iter(p['new_object_SHA256']));oldcache='/refactor/candidates/kimi-compact-cache-r4974/objects/kimi_k3_allocate_cache.o';assert link.count(oldmoe)==1 and link.count(oldcache)==1
 link[link.index(oldmoe)]=str(B/'moe.o');link[link.index(oldcache)]=str(B/'cache.o');native=B/'libKimi_capacity_r170.so';link[link.index('-o')+1]=str(native);link=[('-Wl,-Map,'+str(B/'link.map'))if x.startswith('-Wl,-Map,')else x for x in link];assert '-Wl,--no-undefined'in link
 unchanged={x:pins['all_parent_objects_SHA256'][x]for x in link if x.endswith('.o')and x not in [str(B/'moe.o'),str(B/'cache.o')]};assert len(unchanged)==32
 run(link,'Native_link',120)
 exported=run(['nm','-D','--defined-only','-C',str(native)],'nm',25);assert all(x in exported for x in ['modern_kimi_grouped_I96_r163','modern_kimi_I96_mode_r163','kimi_k3_allocate_cache_tensors'])
 unresolved=run(['nm','-C','--undefined-only',str(B/'cache.o')],'cache_references',20);assert 'Tensor::empty_parameter('in unresolved and 'setDeviceMemoryAsync('in unresolved
 ld=run(['ldd','-r',str(native)],'ldd',45,dict(os.environ,LD_LIBRARY_PATH=str(B)+':'+p['LD_LIBRARY_PATH']));assert 'not found'not in ld and 'undefined symbol:'not in ld
 for path,h in pins['all_parent_objects_SHA256'].items():assert sha(path)==h,path
 q.update(completed=True,parent_CPU_receipt_SHA256=sha(bp),parent_Native_SHA256=p['native_SHA256'],native=str(native),native_SHA256=sha(native),new_object_SHA256={str(B/(n+'.o')):sha(B/(n+'.o'))for n in ['moe','cache']},unchanged_object_SHA256=unchanged,new_MD=deps,source_manifest_SHA256=sha(D/'source-manifest-r170.json'),LD_LIBRARY_PATH=str(B)+':'+p['LD_LIBRARY_PATH'],cache_exact_source_proven=True,scratch_lease_explicit_drain_source_proven=True,scratch_lifetime_actual_execution='UNKNOWN_NOT_RUN',per_I96_call_added_sync=True,performance='UNKNOWN; schedule changed',fullModel_peak='UNKNOWN_STOP')
except BaseException as e:q.update(error_class=type(e).__name__,error=repr(e),traceback=traceback.format_exc())
q['elapsed_s']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','CPU_only','GPU','actual_step','elapsed_s','native_SHA256','error_class']}));assert q['completed']
