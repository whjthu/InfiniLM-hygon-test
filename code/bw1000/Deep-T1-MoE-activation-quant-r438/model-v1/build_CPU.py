"""Two actual Model HOST replacements and exact49object link; no Runtime/dlopen."""
from pathlib import Path
import hashlib,json,os,re,shlex,subprocess,time,traceback
S=Path(__file__).resolve().parent;R=Path('/refactor');B=R/'candidates/Deep-T1-MoE-activation-quant-r438-model-cpu-v1';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();started=time.monotonic()
q=dict(completed=False,passed=False,CPU_only=True,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],publication_commit=os.environ['DEEP438_MODEL_PUBLICATION_COMMIT'],Native_dlopen=False,Runtime_created=False,weights_payload_opened=False,GPU_initialized=False,SDK_initialized=False,Model_loaded=False,commands=[],GPU_GO=False,Model_GO=False)
def save():q['seconds']=time.monotonic()-started;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n')
def run(argv,name,timeout=90):
 q['active_phase']=name;q['commands'].append(argv);save();p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout);(B/(name+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,(name,p.stderr[-6000:],p.stdout[-1500:]);return p.stdout
def deps(text):return set(shlex.split(text.replace(chr(92)+chr(10),' ').split(':',1)[1]))
def fresh(argv):
 a=[x for x in argv if x!='-MD'];i=a.index('-MF');del a[i:i+2];i=a.index('-o');del a[i:i+2];a[a.index('-c')]='-M';return a
def symbols(p,name):
 text=run(['nm','-D','--defined-only',str(p)],name,25);return {x.split()[-1]:x.split()[-2]for x in text.splitlines()if len(x.split())>=3}
try:
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')
 mp=S/'SOURCE_MANIFEST.json';assert sha(mp)==os.environ['DEEP438_MODEL_MANIFEST_SHA'];m=json.loads(mp.read_bytes());assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())
 pin=json.loads((S/'PARENT_PINS.json').read_bytes());assert sha(pin['Native_build'])==pin['Native_build_SHA256']and sha(pin['Native'])==pin['Native_SHA256']
 parent=json.loads(Path(pin['Native_build']).read_bytes())['families']['Deep'];old=parent['Native'];worker=parent['Native_worker'];objects=dict(old['unchanged_objects_SHA256']);objects[worker['new_object']]=worker['new_object_SHA256'];assert len(objects)==48 and all(sha(p)==h for p,h in objects.items())
 parents=[];history={}
 for item in pin['parent_CPU_receipts']:
  assert sha(item['path'])==item['SHA256'];parents.append(json.loads(Path(item['path']).read_bytes()))
 moe=next(x for x in parents[0]['new_compiled_units']if x['source'].endswith('deep_model_owner_r8900.cpp'));bank=next(x for x in parents[1]['compiled_units']if x['source'].endswith('deepseek_v4_int8_mlp_bank_r20.cpp'))
 for entry in [moe,bank]:
  assert sha(entry['source'])==entry['source_SHA256']and objects[entry['object']]==entry['object_SHA256']
  for p,h in entry['deps'].items():assert sha(p)==h;history[p]=h
 assert sha(S/'deepseek_v4_moe_routed_r20.hpp')=='b14a666b2c7d28360a678f9ac348fb10c3e3b436de833939e26b719951ea77e6'
 assert sha(S/'deepseek_v4_int8_mlp_bank_r20.hpp')=='29e9ea71bbf561dd6d4dd2d97d2837ff7342188b254400232b98f33503e9520e'
 for p,v in pin['actual_DTK_bindings'].items():assert sha(p)==v['SHA256']
 compiled=[]
 def compile_one(argv,name):
  before={p:sha(p)for p in deps(run(fresh(argv),name+'-fresh-M',60))}
  for p,h in before.items():
   if Path(p).is_relative_to(S):continue
   if p in history:assert h==history[p],p
   else:assert not p.startswith('/refactor/'),('Unknown project dependency',p)
  run(argv,name+'-compile',100);md=Path(argv[argv.index('-MF')+1]);assert deps(md.read_text())==set(before)and all(sha(p)==h for p,h in before.items())
  obj=argv[argv.index('-o')+1];compiled.append(dict(object=obj,object_SHA256=sha(obj),MD=str(md),MD_SHA256=sha(md),dependencies_SHA256=before,compile_argv=argv,fresh_M_equals_MD=True));return obj
 pure=['g++','-std=c++17','-O2','-pthread','-I'+str(S),'-MD','-MF',str(B/'scope.d'),'-c',str(S/'model_scope_CPU.cpp'),'-o',str(B/'scope.o')];scope=compile_one(pure,'scope');exe=B/'CPU-pure-Model-scope';run(['g++',scope,'-pthread','-o',str(exe)],'scope-link',25)
 dynamic=run(['readelf','-d',str(exe)],'scope-dynamic',20);needed=[line.split('[',1)[1].split(']',1)[0]for line in dynamic.splitlines()if'NEEDED'in line];assert set(needed)<={'libstdc++.so.6','libm.so.6','libgcc_s.so.1','libc.so.6','libpthread.so.0','ld-linux-x86-64.so.2'}
 ld=run(['ldd',str(exe)],'scope-ldd',20);assert'not found'not in ld and not any(v in ld.lower()for v in ['libinfini','libdeep','libtorch','libcuda','libhip','libgalaxy','flash_attn'])
 contract=json.loads(run([str(exe)],'actual-pure-Model-scope-8thread',30).splitlines()[-1]);assert contract['passed']and contract['default_unarmed_OFF']and contract['owner_threads']==8 and not contract['SDK_or_Runtime_executed'];q['actual_scope_CPU_contract']=contract
 consumer=json.loads(run(['python3','-B',str(S/'check_quiescent_CPU.py')],'actual-quiescent-request-consumer',30).splitlines()[-1]);assert consumer['passed']and consumer['actual_qualified_reset_and_thread_methods_consumed']and consumer['Native_ctx_high64_not_truncated'];q['actual_quiescent_CPU_consumer']=consumer
 replacements={}
 for name,entry in [('model_moe',moe),('model_bank',bank)]:
  argv=entry['compile_argv'][:];argv[1:1]=['-I'+str(S),'-I'+str(Path(entry['source']).parent),'-ffp-contract=off'];argv[argv.index('-c')+1]=str(S/(name+'.cpp'));argv[argv.index('-o')+1]=str(B/(name+'.o'));argv[argv.index('-MF')+1]=str(B/(name+'.d'));replacements[entry['object']]=compile_one(argv,name)
 hip=json.loads((S/'REUSED_HIP.json').read_bytes());assert sha(hip['CPU_receipt'])==hip['CPU_receipt_SHA256'];cpu=json.loads(Path(hip['CPU_receipt']).read_bytes());assert cpu['passed']and cpu['actual_CPU']=='910407.889'
 assert sha(hip['object'])==hip['object_SHA256']and sha(hip['MD'])==hip['MD_SHA256'];assert deps(Path(hip['MD']).read_text())==set(hip['deps_SHA256'])and all(sha(p)==h for p,h in hip['deps_SHA256'].items())
 # This is the actual qualified Native graph, not component linking against
 # a still-old Native library. Replace only the two original public classes.
 link=old['link_argv'][:];assert set(x for x in link if x.endswith('.o'))==set(objects)
 link=[replacements.get(x,x)for x in link];first=next(i for i,x in enumerate(link)if x.endswith('.o'));link.insert(first,hip['object']);library=B/'libDeep_T1_MoE_actquant_Model_r438.so';link[link.index('-o')+1]=str(library)
 assert'--no-undefined'in' '.join(link);run(link,'Model49-Native-link-NO-execution',60)
 final={replacements.get(p,p):sha(replacements.get(p,p))for p in objects};final[hip['object']]=hip['object_SHA256'];assert len(final)==49
 oldsyms=symbols(pin['Native'],'old-afc6-defined');newsyms=symbols(library,'new-Model-defined');weak={'W','w','V','v','u'};strong={n for n,k in oldsyms.items()if k not in weak};Cexports={n for n in oldsyms if not n.startswith('_Z')};assert strong<=set(newsyms)and Cexports<=set(newsyms)
 required=['deep_model_actquant_'+n+'_r438'for n in ['set','clear','read','metadata']];assert all(n in newsyms for n in required)
 ff=R/'candidates/deep-owner-device-r8918/deepseek_v4_ffn_routed_block_r20.cpp.o';unresolved=run(['nm','-u',str(ff)],'actual-unchanged-FFN-required-types',20);names={s.split()[-1]for s in unresolved.splitlines()if'DeepseekV4MoERoutedR20'in s};assert len(names)>=3 and names<=set(newsyms)
 assert all('actquant_r438'not in n for n in names),'Original public class is the consumer'
 olddyn=run(['readelf','-d',pin['Native']],'old-afc6-dynamic',20);newdyn=run(['readelf','-d',str(library)],'new-Model-dynamic',20)
 getneeded=lambda t:[x.split('[',1)[1].split(']',1)[0]for x in t.splitlines()if'NEEDED'in x];assert sorted(getneeded(olddyn))==sorted(getneeded(newdyn))
 ld=run(['ldd','-r',str(library)],'new-Model-ldd-r-NO-initializers',30);assert'not found'not in ld and'undefined symbol'not in ld;resolved={n:p for n,p in re.findall(r'^\s*(\S+)\s+=>\s+(\S+)\s+\(',ld,re.M)}
 for p,v in pin['actual_DTK_bindings'].items():assert Path(p).name in resolved and sha(resolved[Path(p).name])==v['SHA256']
 q.update(passed=True,source_manifest_SHA256=sha(mp),library=str(library),library_SHA256=sha(library),production_HOST_compiles=2,pure_scope_HOST_compiles=1,GPU_compiles=0,reused_HIP=dict(object=hip['object'],SHA256=hip['object_SHA256']),compiled=compiled,all_objects_SHA256=final,Native_object_count=49,unchanged_actual_Model_objects=46,all_old_strong_exports_retained=True,all_old_C_exports_retained=True,nonpublic_weak_removed_diagnostic=sorted(set(oldsyms)-set(newsyms)),new_CABI=required,actual_unchanged_FFN_original_MoE_symbols_resolved=sorted(names),DT_NEEDED_unchanged=True,DT_NEEDED=getneeded(newdyn),default_unarmed_OFF=True,Model_Runtime_constructor_executed=False,actual_Model_forward_executed=False,Model_gain='UNKNOWN',CPU_is_not_device_or_Model_qualification=True)
except BaseException as exc:q.update(passed=False,error=repr(exc),traceback=traceback.format_exc())
q['completed']=True;save();print(json.dumps({k:q.get(k)for k in ['passed','actual_CPU','seconds','active_phase','error']}));assert q['passed']
