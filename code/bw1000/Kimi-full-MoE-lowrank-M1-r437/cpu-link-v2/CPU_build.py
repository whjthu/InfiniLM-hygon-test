"""Known883 link repair: exact seven cloned wrapper symbols only, zero compile."""
from pathlib import Path
import hashlib,json,os,struct,subprocess,time,traceback
D=Path(__file__).resolve().parent;R=Path('/refactor');B=R/'candidates/Kimi-full-MoE-lowrank-M1-r437-CPU-link-v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();started=time.monotonic()
q=dict(completed=False,passed=False,CPU_only=True,GPU=0,Native_dlopen=False,weights=0,Runtime_created=False,new_HOST_TUs=0,new_GPU_TUs=0,compile=0,link=1,commands=[])
def run(args,label,cap=35):
 left=260-(time.monotonic()-started);assert left>0
 z=subprocess.run(args,capture_output=True,text=True,timeout=min(cap,left));p=B/(label+'.log');p.write_text(z.stdout+z.stderr);q['commands'].append(dict(argv=args,label=label,rc=z.returncode,log=str(p),SHA256=sha(p)))
 if z.returncode:raise RuntimeError(label+' failed: '+z.stderr[-2500:])
 return z.stdout
def symbols(text):return{line.split()[-1]:line.split()[-2]for line in text.splitlines()if len(line.split())>=3}
def text_sections(path):
 raw=Path(path).read_bytes();header=struct.unpack('<16sHHIQQQIHHHHHH',raw[:64]);assert header[0][:6]==b'\x7fELF\x02\x01'
 offset,width,count,strings=header[6],header[11],header[12],header[13];assert width==64 and count>0
 entries=[struct.unpack('<IIQQQQIIQQ',raw[offset+i*width:offset+(i+1)*width])for i in range(count)]
 table=entries[strings];names=raw[table[4]:table[4]+table[5]];out={}
 for entry in entries:
  name=names[entry[0]:].split(b'\0',1)[0].decode()
  if name.startswith('.text')or entry[2]&4:
   value=raw[entry[4]:entry[4]+entry[5]];out[name]=dict(bytes=len(value),SHA256=hashlib.sha256(value).hexdigest())
 assert out and '.text'in out;return out
def main():
 assert not B.exists(),'Existing artifact: reconcile only';B.mkdir()
 try:
  assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')
  mf=D/'SOURCE_MANIFEST.json';assert sha(mf)==os.environ['KIMI_R437_CPU_LINK_V2_SHA'];m=json.loads(mf.read_bytes());q.update(actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],source_manifest_SHA256=sha(mf))
  for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
  c=json.loads((D/'REUSE_CPU883.json').read_bytes());p=Path(c['receipt']['path']);assert sha(p)==c['receipt']['SHA256'];old=json.loads(p.read_bytes());assert old['actual_CPU']=='910407.883'and not old['passed']and len(old['compiled'])==2
  assert old['source_reverse_contract']['passed']and old['actual_payload_FFI_and_retained_input']['passed'];assert old['commands'][-1]['label']=='isolated-complete-original-plus-private-link'and old['commands'][-1]['rc']==1
  parent=Path(c['parent_source']['path']);assert sha(parent/'SOURCE_MANIFEST.json')==c['parent_source']['SHA256']
  for n,h in json.loads((parent/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(parent/n)==h,n
  for v in old['compiled']:
   assert sha(v['object'])==v['object_SHA256']and sha(v['MD'])==v['MD_SHA256']
   for p,h in v['dependency_SHA256'].items():assert sha(p)==h,p
  contract=json.loads((parent/'BUILD_CONTRACT.json').read_bytes());native=contract['parent_native'];assert sha(native['path'])==native['SHA256']and sha(native['native'])==native['native_SHA256']
  for p,h in native['all_linked_object_SHA256'].items():assert sha(p)==h,p
  g=contract['old_GPU_object'];assert sha(g['object'])==g['object_SHA256']and sha(g['MD'])==g['MD_SHA256']
  for p,h in g['dependency_SHA256'].items():assert sha(p)==h,p
  helper=c['retained_grouped_helper'];assert sha(helper['path'])==helper['SHA256'];assert sha(c['old_link_recipe']['path'])==c['old_link_recipe']['SHA256']
  symbols_helper=symbols(run(['nm','-D','--defined-only',helper['path']],'original-grouped-provider-symbols'))
  for name in ['modern_grouped_mxfp4_owned','modern_grouped_error']:assert symbols_helper.get(name)=='T'
  source=Path(old['compiled'][0]['object']);localized=B/'private_moe.only_7_C_wrappers_local.o'
  exact=c['localize_exact_C_symbols'];before=symbols(run(['nm','-g','--defined-only',str(source)],'before-private-global'))
  assert len(exact)==7 and len(set(exact))==7 and all(before.get(k)=='T'for k in exact)
  run(['objcopy',*['--localize-symbol='+name for name in exact],str(source),str(localized)],'localize-only-seven-clone-wrappers')
  after=symbols(run(['nm','-g','--defined-only',str(localized)],'after-private-global'))
  assert after=={k:v for k,v in before.items()if k not in exact},'Any unrelated global symbol change is forbidden'
  all_after=symbols(run(['nm','--defined-only',str(localized)],'all-localized-symbols'));assert all(all_after.get(k)=='t'for k in exact)
  text_before,text_after=text_sections(source),text_sections(localized);assert text_before==text_after,'Any text-byte change is forbidden'
  q['localized_private_object']=dict(original=str(source),original_SHA256=sha(source),object=str(localized),object_SHA256=sha(localized),only_localized=exact,all_other_global_symbols_unchanged=True,all_text_sections_SHA256=text_before,text_bytes_unchanged=True)
  # Reuse the successful original argv and its exact objects. Only the one copied
  # private object/output/map path and the missing retained provider are changed.
  link=old['commands'][-1]['argv'][:];assert link.count(str(source))==1;link[link.index(str(source))]=str(localized)
  lib=B/'libKimi_full_MoE_lowrank_M1_r437.so';link[link.index('-o')+1]=str(lib)
  link=[('-Wl,-Map,'+str(B/'link.map'))if v.startswith('-Wl,-Map,')else v for v in link]
  at=link.index('-Wl,--no-as-needed')+1;link[at:at]=[helper['path']]
  link.append('-Wl,-rpath,'+str(Path(helper['path']).parent))
  run(link,'private-complete-link-with-retained-provider',60)
  full=symbols(run(['nm','--defined-only',str(lib)],'all-defined'));dyn=symbols(run(['nm','-D','--defined-only',str(lib)],'dynamic-defined'))
  old_export=symbols(run(['nm','-D','--defined-only',native['native']],'old-Native-exports'));required={k:v for k,v in old_export.items()if k.startswith('modern_')and v in ['T','D','B','R']};assert required and all(dyn.get(k)==v for k,v in required.items())
  original_moe=symbols(run(['nm','-g','--defined-only',contract['parent_moe']['object']],'original-strong-MoE'));strong={k for k,v in original_moe.items()if v in ['T','D','B','R']};assert strong<=set(full)
  for name in ['modern_kimi_lowrank_full_moe_r437','modern_kimi_lowrank_full_moe_result_r437','kimi_lowrank_gemv_r432']:assert dyn.get(name)=='T'
  dynamic=run(['readelf','--wide','-d',str(lib)],'actual-dynamic');assert 'SYMBOLIC'in dynamic and 'libkimi_grouped_prefill.so'in dynamic
  ldd=run(['ldd','-r',str(lib)],'actual-ldd-relocations');assert 'not found'not in ldd and 'undefined symbol'not in ldd
  for name,expected in native['actual_dependency_library_SHA256'].items():
   rows=[line.split()for line in ldd.splitlines()if line.split()and line.split()[0]==name];assert len(rows)==1 and rows[0][1]=='=>'and sha(rows[0][2])==expected['SHA256'],name
  rows=[line.split()for line in ldd.splitlines()if line.split()and line.split()[0]=='libkimi_grouped_prefill.so'];assert len(rows)==1 and sha(rows[0][2])==helper['SHA256']
  q.update(completed=True,passed=True,library=str(lib),library_SHA256=sha(lib),old883_receipt=c['receipt'],reused_adapter=old['compiled'][1],reused_original_33_objects_SHA256=native['all_linked_object_SHA256'],reused_GPU_object=g,retained_grouped_helper=helper,old_strong_CABI_preserved=len(required),original_moe_strong_preserved=len(strong),source_reverse_and_consumer_contract_reused=True,CPU883_not_rewritten=True,GPU_numeric_or_TP16_owner_qualification=False)
 except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
 finally:
  q['elapsed_s']=time.monotonic()-started;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['actual_CPU','passed','error','elapsed_s','library_SHA256']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
