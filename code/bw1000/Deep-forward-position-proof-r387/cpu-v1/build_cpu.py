"""Two current HOST TU replacements, no Native dlopen/Runtime/GPU/weights."""
import pathlib,json,hashlib,subprocess,os,time,shlex,re,traceback
R=pathlib.Path('/refactor');S=pathlib.Path(__file__).resolve().parent
B=R/'candidates/Deep-forward-position-proof-r387-cpu-v1';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q=dict(completed=False,passed=False,CPU_only=True,GPU_initialized=False,Native_dlopen=False,
 Runtime_created=False,weights_payload_opened=False,Model_loaded=False,
 actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],commands=[],
 publication_commit=os.environ.get('DEEP387_PUBLICATION_COMMIT','MISSING'))
started=time.perf_counter()
def run(args,label,timeout=100):
 q['commands'].append(args);p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 (B/(label+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,(label,p.returncode,p.stderr[-5000:]);return p.stdout
def parse(t):return set(shlex.split(t.replace('\\\n',' ').split(':',1)[1]))
def symbols(p):return {l.split()[-1]:l.split()[-2]for l in run(['nm','-D','--defined-only',str(p)],'exports-'+p.stem).splitlines()if l.split()}
try:
 assert re.fullmatch('[0-9a-f]{40}',q['publication_commit'])
 mp=S/'SOURCE_MANIFEST.json';assert sha(mp)==os.environ['DEEP387_CPU_MANIFEST_SHA'];m=json.loads(mp.read_bytes())
 assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())
 g=json.loads((S/'PARENT_GRAPH.json').read_bytes());proof=json.loads((S/'SOURCE_PROOF.json').read_bytes())
 p=pathlib.Path(g['parent_CPU_receipt']['path']);assert sha(p)==g['parent_CPU_receipt']['SHA256'];parent=json.loads(p.read_bytes())
 oldlib=pathlib.Path(g['parent_library']);assert sha(oldlib)==g['parent_library_SHA256']==parent['library_SHA256']
 assert parent['passed']and parent['CPU_only']and not parent['GPU_initialized']and not parent['Native_dlopen']
 assert g['all49_objects_SHA256']==parent['all_objects_SHA256']and g['link_argv']==parent['link_argv']
 assert len(g['all49_objects_SHA256'])==49
 for n,h in g['all49_objects_SHA256'].items():assert sha(n)==h
 owner=R/'results/deep-owner-device-CPU-r8918.json';assert sha(owner)==g['decoder_current_receipt_SHA256']
 actual_decoder=next(v for v in json.loads(owner.read_bytes())['new_compiled_units']if v['object']==g['decoder']['object'])
 assert actual_decoder==g['decoder']
 comp=g['compressor'];assert sha(comp['successful_flags_receipt'])==comp['successful_flags_receipt_SHA256']
 units=[];replace={}
 for label in ['decoder','compressor']:
  pr=proof['original_units'][label];original=pathlib.Path(pr['source']);src=S/pr['candidate']
  assert sha(original)==pr['SHA256']and sha(src)==pr['candidate_SHA256']
  stripped=re.sub(rb'/\*POS387_BEGIN_([A-Z]+)\*/.*?/\*POS387_END_\1\*/',b'',src.read_bytes(),flags=re.S)
  assert stripped==original.read_bytes()
  if label=='decoder':
   rec=g['decoder'];oldobj=rec['object'];args=rec['compile_argv'][:rec['compile_argv'].index('-MD')]
  else:oldobj=comp['original_object'];args=comp['flags']
  assert sha(oldobj)==g['all49_objects_SHA256'][oldobj]
  # Original quoted includes resolve from original.parent. Preserve that priority.
  base=[args[0],'-I'+str(original.parent),*args[1:]]
  flags=[args[0],'-I'+str(S),'-I'+str(original.parent),*args[1:]]
  original_deps=parse(run([*base,'-M',str(original)],label+'-original-fresh-M'))
  candidate_deps=parse(run([*flags,'-M',str(src)],label+'-candidate-fresh-M'))
  original_hashes={p:sha(p)for p in original_deps};candidate_hashes={p:sha(p)for p in candidate_deps}
  extra={p:h for p,h in candidate_hashes.items()if p not in original_hashes}
  assert {p:h for p,h in candidate_hashes.items()if p in original_hashes}=={p:h for p,h in original_hashes.items()if p!=str(original)}
  assert not (original_deps-candidate_deps-{str(original)})
  if label=='decoder':
   assert original_hashes==g['decoder']['deps'], 'CurrentdecoderoriginalMDdiff'
  else:
   for p,h in original_hashes.items():
    if p in comp['historical_common_deps']:assert h==comp['historical_common_deps'][p],('CompressorHistoricalCommonDependencyChanged',p)
  obj=B/(label+'.o');md=B/(label+'.d');cmd=[*flags,'-MD','-MF',str(md),'-c',str(src),'-o',str(obj)]
  run(cmd,label+'-compile');assert parse(md.read_text())==candidate_deps
  assert all(sha(p)==h for p,h in candidate_hashes.items())
  units.append(dict(label=label,source=str(src),source_SHA256=sha(src),original_source=str(original),original_source_SHA256=sha(original),object=str(obj),object_SHA256=sha(obj),replaces=oldobj,compile_argv=cmd,dependencies_SHA256=candidate_hashes,original_fresh_M_SHA256=original_hashes,added_dependencies_SHA256=extra,MD_equals_fresh_M=True,original_MD_available=label=='decoder'))
  replace[oldobj]=str(obj)
 link=[replace.get(a,a)for a in g['link_argv']];lib=B/'libDeep_position_proof_r387.so';link[link.index('-o')+1]=str(lib)
 assert len([a for a in link if a.endswith('.o')])==49 and all(a not in link for a in replace)
 run(link,'link',80);olds=symbols(oldlib);news=symbols(lib)
 strong={n for n,t in olds.items()if t.upper()not in ['W','V']};public_C={n for n in olds if not n.startswith('_Z')}
 assert strong<=set(news)and all(news[n].upper()not in ['W','V']for n in strong)
 assert public_C<=set(news)
 weak_delta=[n for n,t in olds.items()if n.startswith('_Z')and t.upper()in ['W','V']and n not in news]
 q['non_public_mangled_weak_export_delta_diagnostic']=sorted(weak_delta)
 required={'deep_position_proof_'+n+'_r387'for n in ['set','get','clear','stats']};assert required<=set(news)
 needed=lambda p:[l.split('[',1)[1].split(']',1)[0]for l in run(['readelf','-d',str(p)],'dynamic-'+p.stem).splitlines()if'NEEDED'in l]
 before=needed(oldlib);after=needed(lib);assert set(before)==set(after)
 ldd=run(['ldd',str(lib)],'ldd');assert 'not found'not in ldd
 for p,v in g['actual_DTK_bindings'].items():assert sha(p)==v['SHA256']
 assert sha(g['existing_HIP_provider']['path'])==g['existing_HIP_provider']['SHA256']
 # Compile the same helper across two HOST translation units against only a mock Runtime.
 mock=B/'mock/runtime';mock.mkdir(parents=True);(mock/'runtime.hpp').write_bytes((S/'CPU_mock_runtime.hpp').read_bytes())
 # Use the exact C ABI block from the actual decoder source, not handwritten replacement exports.
 decoder_source=(S/proof['original_units']['decoder']['candidate']).read_bytes()
 abi=re.search(rb'/\*POS387_BEGIN_CABI\*/(.*?)/\*POS387_END_CABI\*/',decoder_source,flags=re.S).group(1)
 exports=B/'CPU-exact-exports.cpp';exports.write_bytes(b'#include "deep_position_proof_r387.hpp"\n'+abi)
 exe=B/'CPU-contract';run(['g++','-std=c++17','-O0','-UNDEBUG','-pthread','-I'+str(B/'mock'),'-I'+str(S),str(S/'CPU_contract.cpp'),str(S/'CPU_cross_tu.cpp'),str(exports),'-o',str(exe)],'CPU-contract-compile',30)
 contract=json.loads(run([str(exe)],'CPU-contract',10));assert contract['passed']and contract['cross_TU']
 unchanged={p:h for p,h in g['all49_objects_SHA256'].items()if p not in replace};assert len(unchanged)==47
 assert all(sha(p)==h for p,h in unchanged.items())
 objects=dict(unchanged);objects.update({u['object']:u['object_SHA256']for u in units});assert len(objects)==49
 q.update(completed=True,passed=True,source_manifest_SHA256=sha(mp),parent_library_SHA256=sha(oldlib),parent_CPU_receipt_SHA256=sha(pathlib.Path(g['parent_CPU_receipt']['path'])),library=str(lib),library_SHA256=sha(lib),compiled=units,all49_objects_SHA256=objects,unchanged47_objects_SHA256=unchanged,link_argv=link,all_old_strong_and_C_exports_retained=True,new_CABI=sorted(required),original_NEEDED=before,new_NEEDED=after,added_NEEDED=sorted(set(after)-set(before)),removed_NEEDED=sorted(set(before)-set(after)),actual_DTK_bindings=g['actual_DTK_bindings'],existing_HIP_provider=g['existing_HIP_provider'],CPU_contract=contract,default_OFF=True,Model_gain='UNMEASURED',new_Native_Runtime_or_GPU_qualification='NOT_RUN',seconds=time.perf_counter()-started)
except BaseException as exc:q.update(error=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-started)
(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n')
print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_step','library_SHA256','seconds','error']}));assert q['passed']
