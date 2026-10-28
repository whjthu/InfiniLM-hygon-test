"""One HOST MoE compile and Native link from actualCPU517; no HIP/adapter/runtime."""
from pathlib import Path
import ast,json,hashlib,os,subprocess,time,shlex,traceback,sys
R=Path('/refactor');D=Path(__file__).resolve().parent;S=R/'incoming/Kimi-TP32-I96-template-component-r163';B=R/'candidates/Kimi-TP32-I96-exact-storage-r167'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
start=time.monotonic();B.mkdir(exist_ok=False)
q={'completed':False,'CPU_only':True,'GPU':False,'Native_loaded':False,'actual_step':os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),'commands':[],'new_GPU_math':False,'adapter_built':False,'fullModel_GO':False}
def run(cmd,label,seconds,env=None):
 q['commands'].append(cmd);z=subprocess.run(cmd,capture_output=True,text=True,timeout=seconds,env=env);(B/(label+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(label,z.stderr[-2000:]);return z.stdout+z.stderr
try:
 m=json.loads((D/'source-manifest-r167.json').read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 for f in D.glob('*.py'):ast.parse(f.read_bytes())
 pins=json.loads((D/'parent-pins-r167.json').read_bytes());bp=Path(pins['parent_receipt_path']);assert sha(bp)==pins['parent_receipt_SHA256'];p=json.loads(bp.read_bytes())
 assert p['completed']and p['CPU_only']and not p['GPU']and not p['Native_loaded']and p['actual_step']=='910407.517'
 assert sha(p['native'])==p['native_SHA256']==pins['parent_Native_SHA256']
 assert sha(S/'source-manifest-r163.json')==p['source_manifest_SHA256']==pins['parent_source_manifest_SHA256']
 for n,h in json.loads((S/'source-manifest-r163.json').read_bytes())['files_SHA256'].items():assert sha(S/n)==h,n
 for path,h in pins['all_parent_linked_objects_SHA256'].items():assert sha(path)==h,path
 # Check every original MoE dependency, including class-layout headers and creator source.
 for path,h in p['new_MD']['moe.d'].items():assert sha(path)==h,path
 common=R/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/infinilm/_lib/libinfinilm_runtime.so'
 assert sha(common)==pins['Common_Runtime_SHA256']and sha(R/'install/hygon-kimi-5f91ad7-r3672/lib/libinfinirt.so')==pins['RT_SHA256']
 q['source_shape_alias_contract']=json.loads(run([sys.executable,'-B',str(D/'CPU_contract_r167.py'),str(S)],'contract',20))
 cc=[x for x in p['commands']if '-c'in x and x[x.index('-c')+1].endswith('/kimi_moe_I96_r163.cpp')];assert len(cc)==1;cc=cc[0][:]
 cc.insert(1,'-I'+str(D));cc[cc.index('-c')+1]=str(D/'kimi_moe_I96_exact_r167.cpp');cc[cc.index('-o')+1]=str(B/'moe.o');cc[cc.index('-MF')+1]=str(B/'moe.d');run(cc,'HOST_MoE',180)
 native=B/'libKimi_I96_exact_r167.so';link=[x for x in p['commands']if '-shared'in x and '-o'in x and x[x.index('-o')+1]==p['native']];assert len(link)==1;link=link[0][:]
 old=str(R/'candidates/Kimi-TP32-I96-template-r163/moe.o');assert link.count(old)==1;link[link.index(old)]=str(B/'moe.o');link[link.index('-o')+1]=str(native);link=[('-Wl,-Map,'+str(B/'link.map'))if x.startswith('-Wl,-Map,')else x for x in link]
 assert '-Wl,--no-undefined'in link and str(R/'candidates/Kimi-TP32-I96-template-r163/I96.o')in link
 unchanged={x:pins['all_parent_linked_objects_SHA256'][x]for x in link if x.endswith('.o')and x!=str(B/'moe.o')}
 assert len(unchanged)==33 and len([x for x in link if x.endswith('.o')])==34
 run(link,'Native_link',120)
 names=shlex.split((B/'moe.d').read_text().replace('\\\n',' ').split(':',1)[1]);deps={n:sha(n)for n in names if Path(n).is_file()}
 header=str(R/'candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/models/kimi_k3/kimi_k3_moe.hpp');assert deps[header]==pins['critical_header_SHA256']
 for path,h in p['new_MD']['moe.d'].items():
  if path.endswith(('.h','.hpp','.cuh')):assert deps[path]==h,path
 symbols=run(['nm','-D','--defined-only',str(native)],'Native_nm',20)
 for name in ['modern_kimi_grouped_I96_r163','modern_kimi_I96_mode_r163','modern_kimi_I96_counts_r163']:assert name in symbols,name
 undefined=run(['nm','-C','--undefined-only',str(B/'moe.o')],'MoE_undefined_regular',20);assert 'infinilm::runtime::Tensor::empty_parameter('in undefined
 ld=run(['ldd','-r',str(native)],'Native_ldd',45,dict(os.environ,LD_LIBRARY_PATH=str(B)+':'+p['LD_LIBRARY_PATH']));assert 'not found'not in ld and 'undefined symbol:'not in ld
 for path,h in unchanged.items():assert sha(path)==h,path
 q.update(completed=True,parent_receipt_path=str(bp),parent_receipt_SHA256=sha(bp),parent_actual_step=p['actual_step'],parent_native_SHA256=p['native_SHA256'],native=str(native),native_SHA256=sha(native),new_object_SHA256={str(B/'moe.o'):sha(B/'moe.o')},unchanged_object_SHA256=unchanged,new_MD={'moe.d':deps},layer_headers=p['layer_headers'],LD_LIBRARY_PATH=str(B)+':'+p['LD_LIBRARY_PATH'],source_manifest_SHA256=sha(D/'source-manifest-r167.json'),math_change='NONE; only four packed persistent storage allocation APIs',actual_GPU_memory_saving='UNKNOWN',fullModel_peak='UNKNOWN_STOP',adapter_note='No adapter built: old r163 adapter binds old Native and must not be used to claim this candidate qualified')
except BaseException as e:q.update(error_class=type(e).__name__,error=repr(e),traceback=traceback.format_exc())
q['elapsed_s']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','CPU_only','GPU','actual_step','elapsed_s','native_SHA256','error_class']}));assert q['completed']
