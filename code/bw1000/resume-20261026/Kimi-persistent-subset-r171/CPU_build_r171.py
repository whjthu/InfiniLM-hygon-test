"""One additional HOST MoE+probe TU; actual r170 cache/other objects reused."""
from pathlib import Path
import ast,json,hashlib,os,subprocess,time,shlex,sys,traceback
R=Path('/refactor');D=Path(__file__).resolve().parent;B=R/'candidates/Kimi-persistent-subset-r171';B.mkdir(exist_ok=False);start=time.monotonic();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q={'completed':False,'CPU_only':True,'GPU':False,'Native_loaded':False,'GPU_GO':False,'fullModel_GO':False,'actual_step':os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),'commands':[]}
def run(a,label,seconds,env=None):
 q['commands'].append(a);p=subprocess.run(a,text=True,capture_output=True,timeout=seconds,env=env);(B/(label+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,(label,p.stderr[-2500:]);return p.stdout+p.stderr
try:
 m=json.loads((D/'source-manifest-r171.json').read_bytes());assert m['GPU_GO']is False
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 for f in D.glob('*.py'):ast.parse(f.read_bytes())
 pins=json.loads((D/'parent-pins-r171.json').read_bytes());bp=Path(pins['parent_CPU_path']);assert sha(bp)==pins['parent_CPU_SHA256'];p=json.loads(bp.read_bytes());assert p['completed']and p['CPU_only']and not p['GPU']and not p['Native_loaded']and p['actual_step']=='910407.533'and sha(p['native'])==p['native_SHA256']==pins['parent_Native_SHA256']
 parent=Path(pins['parent_stage']);assert sha(parent/'source-manifest-r170.json')==p['source_manifest_SHA256']==pins['parent_source_manifest_SHA256']
 for n,h in json.loads((parent/'source-manifest-r170.json').read_bytes())['files_SHA256'].items():assert sha(parent/n)==h,n
 for obj,h in pins['all_parent_objects_SHA256'].items():assert sha(obj)==h,obj
 for deps in pins['parent_MD'].values():
  for name,h in deps.items():assert sha(name)==h,name
 validator=json.loads((D/'cache-validator-source-r171.json').read_bytes());assert sha(validator['actual_source_path'])==validator['actual_source_SHA256'],'cache validator active source differs: stop before any GPU'
 q['CPU_contract']=json.loads(run([sys.executable,'-B',str(D/'CPU_contract_r171.py'),str(parent)],'contract',20))
 cc=[a for a in p['commands']if '-c'in a and a[a.index('-c')+1].endswith('/kimi_moe_capacity_r170.cpp')];assert len(cc)==1;cc=cc[0][:];cc.insert(1,'-I'+str(D));cc[cc.index('-c')+1]=str(D/'probe_r171.cpp');cc[cc.index('-o')+1]=str(B/'moe_probe.o');cc[cc.index('-MF')+1]=str(B/'moe_probe.d');run(cc,'HOST_probe',240)
 link=[a for a in p['commands']if '-shared'in a and '-o'in a and a[a.index('-o')+1]==p['native']];assert len(link)==1;link=link[0][:];old=str(Path(p['native']).parent/'moe.o');assert link.count(old)==1;link[link.index(old)]=str(B/'moe_probe.o');native=B/'libKimi_persistent_subset_r171.so';link[link.index('-o')+1]=str(native);link=[('-Wl,-Map,'+str(B/'link.map'))if x.startswith('-Wl,-Map,')else x for x in link];assert '-Wl,-Bsymbolic'in link and '-Wl,--no-undefined'in link and pins['cache_object']in link
 unchanged={x:pins['all_parent_objects_SHA256'][x]for x in link if x.endswith('.o')and x!=str(B/'moe_probe.o')};assert len(unchanged)==33
 run(link,'Native_link',120);nm=run(['nm','-D','--defined-only',str(native)],'Native_nm',25);assert 'modern_kimi_persistent_subset_r171'in nm and 'modern_kimi_persistent_subset_result_r171'in nm
 cache_symbols=[l.split()[-1]for l in nm.splitlines()if 'kimi_k3_allocate_cache_tensors'in l and '.cold'not in l];assert len(cache_symbols)==1
 paths=shlex.split((B/'moe_probe.d').read_text().replace(chr(92)+chr(10),' ').split(':',1)[1]);deps={n:sha(n)for n in paths if Path(n).is_file()};assert deps[str(parent/'I96_scratch_lease_r170.hpp')]==sha(parent/'I96_scratch_lease_r170.hpp')
 ld=run(['ldd','-r',str(native)],'ldd',45,dict(os.environ,LD_LIBRARY_PATH=str(B)+':'+p['LD_LIBRARY_PATH']));assert 'not found'not in ld and 'undefined symbol:'not in ld
 for obj,h in pins['all_parent_objects_SHA256'].items():assert sha(obj)==h,obj
 q.update(completed=True,parent_CPU_SHA256=sha(bp),parent_Native_SHA256=p['native_SHA256'],native=str(native),native_SHA256=sha(native),probe_source_SHA256=sha(D/'probe_r171.cpp'),production_prefix_SHA256=sha(D/'probe-prefix-r171.inc'),cache_object_reused_SHA256=sha(pins['cache_object']),cache_factory_symbol=cache_symbols[0],new_object_SHA256={str(B/'moe_probe.o'):sha(B/'moe_probe.o')},unchanged_object_SHA256=unchanged,new_MD=deps,source_manifest_SHA256=sha(D/'source-manifest-r171.json'),LD_LIBRARY_PATH=str(B)+':'+p['LD_LIBRARY_PATH'],actual_GPU_validation='NOT_RUN',fullModel_peak='UNKNOWN_STOP')
except BaseException as e:q.update(error_class=type(e).__name__,error=repr(e),traceback=traceback.format_exc())
q['elapsed_s']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','CPU_only','GPU','GPU_GO','actual_step','elapsed_s','native_SHA256','error_class']}));assert q['completed']
