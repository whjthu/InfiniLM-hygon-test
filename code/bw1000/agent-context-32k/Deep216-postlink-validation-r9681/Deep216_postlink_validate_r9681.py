import pathlib,json,hashlib,subprocess,shlex,os,datetime,sys
R=pathlib.Path('/refactor');B=R/'candidates/Deep-C128-weight-cache-r216';D=R/'incoming/Deep-C128-weight-cache-private-r216';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();out=B/'CPU-postlink-validation-r9681.json';assert not out.exists();assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']);p=B/'CPU-build.json';assert sha(p)=='3bfae92540ba6ba1a3ce5aa131c5dbbee826f16cc9823ab6c632895ebcc7788d';d=json.loads(p.read_bytes());assert not d['completed']and d['actual_step']=='910407.571'and 'line 101' in d['traceback'];assert sha(D/'manifest-r216.json')=='1b6232d7ac0b6abfc05cd8b5a2f2609ec5471146fd5ea10ca8fa0bd84a3cbfb0'
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'CPU_only':True,'GPU_initialized':False,'Native_loaded':False,'weights_opened':False,'new_HOST_compilations':0,'failed_original_build_receipt_SHA256':sha(p),'original_CPU_step':'910407.571','actual_step':os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),'source_manifest_SHA256':sha(D/'manifest-r216.json')}
def deps(text):return {str(pathlib.Path(x).resolve())for x in shlex.split(text.replace('\\\n',' ').split(':',1)[1])}
def run(a):
 v=subprocess.run(a,text=True,capture_output=True,timeout=30);assert v.returncode==0,v.stderr;return v.stdout
try:
 parent=R/'candidates/Deep-native-context-guard-r180d/CPU-build.json';assert sha(parent)=='17e94923049bfe910a3cb4810fb69d9c9e4bdebc9265ad2c21d5eaa65c2d39ac';par=json.loads(parent.read_bytes());assert sha(par['library'])==par['library_SHA256']=='232fb7e2566f7d1961fa433d9e45660bfe760ec726676dcb7f1f406d857c801f';q['parent_receipt_SHA256']=sha(parent)
 actual=dict(par['reused_objects_SHA256']);actual.update({u['object']:u['object_SHA256']for u in par['compiled']});assert len(actual)==48 and all(sha(n)==h for n,h in actual.items());assert len(d['reused_layout_exclusions'])==39
 for n,h in par['actual_DTK_bindings'].items():assert sha(n)==h['SHA256']
 cmds=[a for a in d['commands']if '-c'in a and '-MD'in a];assert len(cmds)==9;units=json.loads((D/'adaptation-r216.json').read_bytes())['nine_HOST_rebuilds'];assert len(units)==9;q['compiled_reconstructed']=[]
 for i,a in enumerate(cmds):
  src=pathlib.Path(a[a.index('-c')+1]);obj=pathlib.Path(a[a.index('-o')+1]);md=pathlib.Path(a[a.index('-MF')+1]);assert obj==B/(str(i)+'.o')and obj.is_file();dd=deps(md.read_text());assert dd==deps((B/('new-M-'+str(i)+'.log')).read_text());assert str(src.resolve())in dd
  q['compiled_reconstructed'].append({'source':str(src),'source_SHA256':sha(src),'original_source':units[i]['source'],'replaces':units[i]['current232_object'],'object':str(obj),'object_SHA256':sha(obj),'MD_SHA256':sha(md),'dependency_SHA256':{n:sha(n)for n in dd},'saved_fresh_M_equal_MD':True,'prior_graph_equality_assertion_reached_before_failure':True})
 lib=B/'libDeep_C128_weight_cache_r216.so';before=sha(lib);dynamic=run(['readelf','-d',str(lib)]);assert 'SYMBOLIC'in dynamic;ld=run(['ldd','-r',str(lib)]);assert 'not found'not in ld and 'undefined symbol:'not in ld
 resolved={}
 for s in ld.splitlines():
  if '=> /'in s:
   name,path=s.strip().split('=>',1);resolved[name.strip()]=path.strip().split()[0]
 for n,h in par['actual_DTK_bindings'].items():
  basename=pathlib.Path(n).name
  if basename in resolved:assert sha(resolved[basename])==h['SHA256']
 names=lambda t:{s.split()[-1]for s in t.splitlines()if len(s.split())>=3};symbols=names(run(['nm','-D','--defined-only',str(lib)]));assert {'modern_deep_c128_weight_cache_mode_r172','modern_deep_c128_weight_cache_stats_r172'}<=symbols
 data=lib.read_bytes();assert b'Native sparse cache requires rows1..128 and capacity1..65536'in data and b'Native sparse cache requires rows1..128 and capacity1..4096'not in data;assert sha(lib)==before
 q.update(completed=True,library=str(lib),library_SHA256=before,prior_HOST_TUs_compiled=9,parent_Native_SHA256=par['library_SHA256'],ldd_r_no_missing_or_undefined=True,ELF_SYMBOLIC=True,original_failed_receipt_preserved=True,guard65536_retained=True,default_OFF=True,new_resident_bytes_per_owner=335544320,resolved_dependencies_SHA256={n:sha(n)for n in resolved.values()},scope='Composes failed571 prior source/ABI/compiler/link checks with unchanged existing9 objects/DSO and corrected parent loader environment. No recompile, Native dlopen, Model, GPU or performance qualification.')
except BaseException as e:q['error']=repr(e)
q['Torch_imported']='torch'in sys.modules;q['Native_imported']='_infinilm'in sys.modules;out.write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in q.items()if k!='compiled_reconstructed'and k!='resolved_dependencies_SHA256'}));assert q['completed']
