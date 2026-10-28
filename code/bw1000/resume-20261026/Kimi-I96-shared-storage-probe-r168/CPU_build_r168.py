"""Two HOST MoE+probe TUs and Native links. Never imports/loads GPU code."""
from pathlib import Path
import ast,json,hashlib,os,subprocess,time,shlex,traceback,sys
R=Path('/refactor');D=Path(__file__).resolve().parent;S=R/'incoming/Kimi-TP32-I96-template-component-r163';B=R/'candidates/Kimi-I96-shared-storage-probe-r168';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
q={'completed':False,'CPU_only':True,'GPU':False,'Native_loaded':False,'actual_step':os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),'commands':[],'arms':{},'fullModel_GO':False}
def run(cmd,label,seconds,env=None):
 q['commands'].append(cmd);z=subprocess.run(cmd,capture_output=True,text=True,timeout=seconds,env=env);(B/(label+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(label,z.stderr[-2000:]);return z.stdout+z.stderr
try:
 for f in D.glob('*.py'):ast.parse(f.read_bytes())
 m=json.loads((D/'source-manifest-r168.json').read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 pins=json.loads((D/'parent-pins-r168.json').read_bytes());bp=Path(pins['parent_receipt_path']);assert sha(bp)==pins['parent_receipt_SHA256'];p=json.loads(bp.read_bytes());assert p['completed']and p['CPU_only']and not p['GPU']and not p['Native_loaded']and p['actual_step']=='910407.517'
 assert sha(p['native'])==p['native_SHA256']==pins['parent_Native_SHA256']
 ep=Path(pins['actual_r167_CPU_receipt_path']);assert sha(ep)==pins['actual_r167_CPU_receipt_SHA256'];e=json.loads(ep.read_bytes());assert e['completed']and e['CPU_only']and not e['GPU']and not e['Native_loaded']and e['actual_step']==pins['actual_r167_CPU_step'];assert sha(e['native'])==e['native_SHA256']==pins['actual_r167_Native_SHA256']
 assert sha(R/'incoming/Kimi-TP32-I96-exact-storage-r167/kimi_moe_I96_exact_r167.cpp')==pins['parent_exact_source_SHA256']
 assert sha(S/'source-manifest-r163.json')==pins['parent_source_manifest_SHA256']
 for n,h in json.loads((S/'source-manifest-r163.json').read_bytes())['files_SHA256'].items():assert sha(S/n)==h,n
 for path,h in pins['all_parent_linked_objects_SHA256'].items():assert sha(path)==h,path
 for path,h in p['new_MD']['moe.d'].items():assert sha(path)==h,path
 common=R/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/infinilm/_lib/libinfinilm_runtime.so';assert sha(common)==pins['Common_Runtime_SHA256'];assert sha(R/'install/hygon-kimi-5f91ad7-r3672/lib/libinfinirt.so')==pins['RT_SHA256']
 q['CPU_source_contract']=json.loads(run([sys.executable,'-B',str(D/'CPU_contract_r168.py'),str(S)],'contract',20))
 compile_parent=[x for x in p['commands']if '-c'in x and x[x.index('-c')+1].endswith('/kimi_moe_I96_r163.cpp')];assert len(compile_parent)==1
 link_parent=[x for x in p['commands']if '-shared'in x and '-o'in x and x[x.index('-o')+1]==p['native']];assert len(link_parent)==1
 for arm in ['regular','exact']:
  A=B/arm;A.mkdir(exist_ok=False);obj=A/'moe_probe.o';native=A/('libKimi_I96_storage_'+arm+'_r168.so');cc=compile_parent[0][:];cc.insert(1,'-I'+str(D));cc[cc.index('-c')+1]=str(D/('kimi_moe_storage_'+arm+'_r168.cpp'));cc[cc.index('-o')+1]=str(obj);cc[cc.index('-MF')+1]=str(A/'moe_probe.d');run(cc,arm+'_HOST',180)
  link=link_parent[0][:];old=str(R/'candidates/Kimi-TP32-I96-template-r163/moe.o');assert link.count(old)==1;link[link.index(old)]=str(obj);link[link.index('-o')+1]=str(native);link=[('-Wl,-Map,'+str(A/'link.map'))if x.startswith('-Wl,-Map,')else x for x in link];assert '-Wl,--no-undefined'in link
  unchanged={x:pins['all_parent_linked_objects_SHA256'][x]for x in link if x.endswith('.o')and x!=str(obj)};assert len(unchanged)==33
  run(link,arm+'_link',120)
  nm=run(['nm','-D','--defined-only',str(native)],arm+'_nm',20);assert all(s in nm for s in ['modern_kimi_I96_storage_probe_r168','modern_kimi_I96_storage_probe_result_r168'])
  names=shlex.split((A/'moe_probe.d').read_text().replace(chr(92)+chr(10),' ').split(':',1)[1]);deps={n:sha(n)for n in names if Path(n).is_file()};header=str(R/'candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/models/kimi_k3/kimi_k3_moe.hpp');assert deps[header]==pins['critical_header_SHA256']
  for path,h in p['new_MD']['moe.d'].items():
   if path.endswith(('.h','.hpp','.cuh')):assert deps[path]==h,path
  ld=run(['ldd','-r',str(native)],arm+'_ldd',45,dict(os.environ,LD_LIBRARY_PATH=str(A)+':'+p['LD_LIBRARY_PATH']));assert 'not found'not in ld and 'undefined symbol:'not in ld
  q['arms'][arm]={'native':str(native),'native_SHA256':sha(native),'object':str(obj),'object_SHA256':sha(obj),'source_SHA256':sha(D/('kimi_moe_storage_'+arm+'_r168.cpp')),'new_MD':deps,'unchanged_objects_SHA256':unchanged}
 for path,h in pins['all_parent_linked_objects_SHA256'].items():assert sha(path)==h,path
 q.update(completed=True,parent_receipt_SHA256=sha(bp),parent_native_SHA256=p['native_SHA256'],source_manifest_SHA256=sha(D/'source-manifest-r168.json'),LD_LIBRARY_PATH=p['LD_LIBRARY_PATH'],shared_helper_source_equivalence=True,actual_GPU_memory_delta='UNKNOWN',fullModel_peak='UNKNOWN_STOP',no_comm_no_fake_world=True)
except BaseException as e:q.update(error_class=type(e).__name__,error=repr(e),traceback=traceback.format_exc())
q['elapsed_s']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','CPU_only','GPU','actual_step','elapsed_s','error_class']}));assert q['completed']
