"""One actual TextModel HOST replacement + pure CPU control consumer; no dlopen."""
from pathlib import Path
import ast,hashlib,json,os,re,shlex,subprocess,time,traceback
S=Path(__file__).resolve().parent
B=Path('/refactor/candidates/Deep-layer2-3-boundary-diagnostic-r447-CPU-v2')
B.mkdir(exist_ok=False)
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
start=time.monotonic()
q=dict(completed=False,passed=False,CPU_only=True,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
 publication_commit=os.environ['DEEP447_PUBLICATION_COMMIT'],source_manifest_SHA256=os.environ['DEEP447_MANIFEST_SHA'],commands=[],
 GPU=0,GPU_GO=False,Model_GO=False,Native_dlopen=False,Runtime_created=False,SDK_executed=False,weights_payload_opened=False)
def save():q['seconds']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n')
def run(argv,name,timeout=70):
 q['active_phase']=name;q['commands'].append(argv);save()
 p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
 (B/(name+'.log')).write_text(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError((name,p.returncode,p.stderr[-4500:],p.stdout[-1000:]))
 return p.stdout
def dependencies(text):return set(shlex.split(text.replace(chr(92)+chr(10),' ').split(':',1)[1]))
def fresh(argv):
 a=[v for v in argv if v!='-MD'];i=a.index('-MF');del a[i:i+2];i=a.index('-o');del a[i:i+2]
 if '-c'in a:a[a.index('-c')]='-M'
 else:a.insert(-1,'-M')
 return a
def needed(text):return sorted(x.split('[',1)[1].split(']',1)[0]for x in text.splitlines()if'NEEDED'in x)
def symbols(path,name):
 text=run(['nm','-D','--defined-only',str(path)],name,25)
 return {s.split()[-1]:s.split()[-2]for s in text.splitlines()if len(s.split())>=3}
try:
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')
 assert H(S/'SOURCE_MANIFEST.json')==q['source_manifest_SHA256']
 manifest=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(H(S/n)==h for n,h in manifest['files_SHA256'].items())
 pin=json.loads((S/'PARENT_PINS.json').read_bytes());inputs=json.loads((S/'BUILD_INPUTS.json').read_bytes())
 assert H(pin['parent_CPU_receipt'])==pin['parent_CPU_receipt_SHA256']
 parent=json.loads(Path(pin['parent_CPU_receipt']).read_bytes())
 assert parent['passed']and parent['actual_CPU']=='910407.911'and parent['Native_object_count']==49
 assert parent['all_objects_SHA256']==pin['objects']and H(parent['library'])==pin['parent_Native_SHA256']==parent['library_SHA256']
 assert all(H(p)==h for p,h in pin['objects'].items())and H(pin['original_source'])==pin['original_source_SHA256']
 prior=Path('/refactor/incoming/Deep-T1-MoE-activation-quant-r438/model-diagnostic-v2')
 assert H(prior/'SOURCE_MANIFEST.json')==pin['parent_source_manifest_SHA256']
 assert all(H(prior/n)==h for n,h in pin['old_frozen_modules'].items())
 # Preserve old recorded paths AND hashes. The new explicit metadata map only
 # looks up identical historical mirror bytes at their original parent paths;
 # never edit old pins, source files or use a broad path/unknown-dep waiver.
 history={}
 for item in inputs['project_dependency_identity_records']:
  p,h=item['lookup_path'],item['SHA256']
  if p in history:assert history[p]==h
  history[p]=h
 # Eight original class headers are independently pinned by the historical
 # pre-PD inventory. The PD mirror bytes are a separate friend-declaration
 # domain. Do not replace either historical SHA or accept a fresh hash alone.
 inventory=S/'legacy-original-header-inventory.py'
 assert H(inventory)==inputs['original_header_inventory_proof']['historical_fetch_source_SHA256']
 tree=ast.parse(inventory.read_bytes());refs=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='refs'for t in n.targets)))
 for item in inputs['original_domain_override_records']:
  original=next(r for r in refs if r['path']==item['original_recorded_inventory_path'])
  assert original['SHA256']==item['original_SHA256']
  assert history[item['path']]==item['PD_modified_SHA256']
  assert H(S/item['original_snapshot'])==item['original_SHA256']
  history[item['path']]=item['original_SHA256']
 for p,v in inputs['actual_DTK_bindings'].items():assert H(p)==v['SHA256']
 def compile_checked(argv,name):
  before={p:H(p)for p in dependencies(run(fresh(argv),name+'-fresh-M',55))}
  for p,h in before.items():
   if Path(p).is_relative_to(S):continue
   if p.startswith('/refactor/'):assert p in history and history[p]==h,('Unpinned or changed original project header',p)
  run(argv,name+'-compile',95)
  md=Path(argv[argv.index('-MF')+1]);assert dependencies(md.read_text())==set(before)and all(H(p)==h for p,h in before.items())
  return dict(object=argv[argv.index('-o')+1],object_SHA256=H(argv[argv.index('-o')+1]),MD=str(md),MD_SHA256=H(md),dependencies_SHA256=before,compile_argv=argv,fresh_M_equals_MD=True)
 # Reuse ONLY the original919 completed recording phase;919 overallFAIL
 # remains false. No repeated184forward/old911 recording cases in this job.
 reuse=json.loads((S/'REUSE_CPU919_RECORDING.json').read_bytes())
 assert H(reuse['receipt'])==reuse['receipt_SHA256']and H(reuse['recording_log'])==reuse['recording_log_SHA256']
 failed=json.loads(Path(reuse['receipt']).read_bytes())
 assert failed['completed']and not failed['passed']and failed['actual_CPU']=='910407.919'
 local=failed['actual_changed_consumer'];assert local['completed']and local['passed']and local['actual_Native_execution']is False and local['GPU']==0
 assert [(c['name'],c['completed_forwards'],c['attempts'])for c in local['cases']]==[('normal',184,184),('forward_nonzero',11,12),('continuity_difference',91,91)]
 q['reused_changed_consumer_from_CPU919']=local
 q['changed_recording_executed_this_job']=False
 base=pin['original_exact_source_parent_M_argv'][:]
 assert base[-2:]==['-M',pin['original_source']]
 base[1:1]=['-I'+str(S),'-I'+str(Path(pin['original_source']).parent)]
 base[base.index('-M')]='-c';base[-1]=str(S/'text_model.cpp')
 argv=base+['-MD','-MF',str(B/'text_model.d'),'-o',str(B/'text_model.o')]
 production=compile_checked(argv,'actual-TextModel-HOST')
 # Test the very same header CPU-byte/control functions without creating any
 # Runtime or linking SDK. GPU-facing inline save methods are not called.
 pure=[x for x in base if not x.startswith(('-L','-l'))]
 pure.remove('-c');pure[pure.index(str(S/'text_model.cpp'))]=str(S/'check_control_CPU.cpp')
 pure+=['-MD','-MF',str(B/'control.d'),'-o',str(B/'control')]
 recorded=compile_checked(pure,'pure-CPU-control')
 pure_dyn=run(['readelf','-d',str(B/'control')],'pure-control-DT_NEEDED',20)
 assert set(needed(pure_dyn))<={'libstdc++.so.6','libgcc_s.so.1','libc.so.6','libm.so.6','libpthread.so.0'}
 q['actual_header_control_recording']=json.loads(run([str(B/'control')],'actual-header-control-no-Runtime',15).splitlines()[-1])
 assert q['actual_header_control_recording']['passed']and q['actual_header_control_recording']['Runtime_created']is False
 link=pin['original_actual911_link_argv'][:]
 assert set(x for x in link if x.endswith('.o'))==set(pin['objects'])and pin['replace_object']in link
 link=[str(B/'text_model.o')if x==pin['replace_object']else x for x in link]
 library=B/'libDeep_layer_boundary_r447.so';link[link.index('-o')+1]=str(library)
 assert'--no-undefined'in' '.join(link);run(link,'actual49object-link-NO-execution',55)
 final={str(B/'text_model.o')if p==pin['replace_object']else p:H(B/'text_model.o')if p==pin['replace_object']else h for p,h in pin['objects'].items()}
 assert len(final)==49 and sum(pin['objects'].get(p)==h for p,h in final.items())==48
 oldsyms=symbols(parent['library'],'parent911-defined');newsyms=symbols(library,'r447-defined');weak={'W','w','V','v','u'}
 strong={n for n,k in oldsyms.items()if k not in weak};c_exports={n for n in oldsyms if not n.startswith('_Z')}
 assert strong<=set(newsyms)and c_exports<=set(newsyms)
 added=['deep_boundary_'+n+'_r447'for n in ['arm','seal','abandon','clear','count','info','copy']];assert all(n in newsyms for n in added)
 olddyn=run(['readelf','-d',parent['library']],'parent911-dynamic',20);newdyn=run(['readelf','-d',str(library)],'r447-dynamic',20)
 assert needed(olddyn)==needed(newdyn)
 ld=run(['ldd','-r',str(library)],'r447-ldd-r-NO-initializers',25);assert'not found'not in ld and'undefined symbol'not in ld
 resolved={n:p for n,p in re.findall(r'^\s*(\S+)\s+=>\s+(\S+)\s+\(',ld,re.M)}
 for p,v in inputs['actual_DTK_bindings'].items():assert Path(p).name in resolved and H(resolved[Path(p).name])==v['SHA256']
 q.update(passed=True,library=str(library),library_SHA256=H(library),production_HOST_compiles=1,pure_CPU_control_HOST_compiles=1,HIP_compiles=0,
  compiled=[production,recorded],all_objects_SHA256=final,Native_object_count=49,reused_objects=48,all_old_strong_retained=True,all_old_C_exports_retained=True,
  nonpublic_weak_removed_diagnostic=sorted(set(oldsyms)-set(newsyms)),new_CABI=added,DT_NEEDED_unchanged=True,DT_NEEDED=needed(newdyn),default_unarmed_OFF=True,
  actual_Model_forward_executed=False,actual_Model_activation_captured=False,original911_suites_replayed=False,root_cause='UNKNOWN',CPU_not_device_qualification=True)
except BaseException as error:q.update(passed=False,error=repr(error),traceback=traceback.format_exc())
q['completed']=True;save();print(json.dumps({k:q.get(k)for k in ['passed','actual_CPU','seconds','active_phase','error']}));assert q['passed']
