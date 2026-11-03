"""Two necessary HOST TUs plus pure changed lease/span contract. No SDK execution."""
from pathlib import Path
import hashlib,json,os,shlex,struct,subprocess,time,traceback
D=Path(__file__).resolve().parent;B=Path('/refactor/candidates/Kimi-lowrank-TP-rowshard-r440-CPU-source-span-v2');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
q=dict(passed=False,completed=False,GPU=0,Native_dlopen=False,Runtime_created=False,weights=0,new_production_HOST_TUs=2,pure_contract_HOST_TUs=1,new_GPU_TUs=0,old_successful_TUs_recompiled=False,commands=[])
def run(a,n,cap=45):
 left=260-(time.monotonic()-start);assert left>0
 z=subprocess.run(a,capture_output=True,text=True,timeout=min(left,cap));p=B/(n+'.log');p.write_text(z.stdout+z.stderr);q['commands'].append(dict(argv=a,label=n,rc=z.returncode,log=str(p),SHA256=sha(p)))
 if z.returncode:raise RuntimeError(n+': '+z.stderr[-2500:])
 return z.stdout
def symbols(s):return {v.split()[-1]:v.split()[-2]for v in s.splitlines()if len(v.split())>=3}

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
 assert not B.exists();B.mkdir()
 try:
  assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')
  m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert sha(D/'SOURCE_MANIFEST.json')==os.environ['KIMI_R440_SPAN_V2_SHA']
  for p,h in m['files_SHA256'].items():assert sha(D/p)==h,p
  q.update(actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],entry_affinity=sorted(os.sched_getaffinity(0)),source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'));assert len(q['entry_affinity'])>=4
  refs=json.loads((D/'SOURCE_REFERENCES.json').read_bytes());references={}
  for entry in refs['retained_receipts']:
   p=entry['path'];assert sha(p)==entry['SHA256'],p
   v=json.loads(Path(p).read_bytes());assert v['passed'];
   for c in v['compiled']:
    assert sha(c['object'])==c['object_SHA256']and sha(c['MD'])==c['MD_SHA256']
    for p,h in c['dependency_SHA256'].items():assert sha(p)==h;references[p]=h
  prior=json.loads(Path(refs['retained_receipts'][0]['path']).read_bytes())
  root900=json.loads(Path(refs['CPU900']['path']).read_bytes());assert sha(refs['CPU900']['path'])==refs['CPU900']['SHA256']and root900['passed']
  for p,h in root900['actual_production_compile']['dependency_SHA256'].items():assert sha(p)==h;references[p]=h
  for p,h in refs['explicit_actual_sources'].items():assert sha(p)==h;references[p]=h
  c=json.loads((D/'BUILD_CONTRACT.json').read_bytes());native=c['parent_native'];assert sha(native['path'])==native['SHA256']and sha(native['native'])==native['native_SHA256']
  for p,h in native['all_linked_object_SHA256'].items():assert sha(p)==h
  q['actual_source_contract']=json.loads(run([__import__('sys').executable,'-B',str(D/'check_source_contract.py')],'changed-actual-full-consumer-source-contract',15))
  exe=B/'check_contract';run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(D),str(D/'check_contract.cpp'),'-o',str(exe)],'pure-changed-contract-compile',20)
  q['actual_pure_contract']=json.loads(run([str(exe)],'pure-live-refs-completion-and-single-row-span',15));assert q['actual_pure_contract']['passed']
  q['compiled']=[]
  for name in ['private_moe.cpp','component.cpp']:
   old=next(v for v in prior['commands']if v['label']==name+'-compile')['argv'];prefix=old[:old.index('-MD')];prefix.insert(1,'-I'+str(D));obj=B/(name+'.o');md=B/(name+'.d');pre=B/(name+'.pre.d')
   run(prefix+['-M','-MF',str(pre),'-MT',str(obj),str(D/name)],name+'-preMD',30)
   def deps(p):return set(shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1]))
   before={p:sha(p)for p in deps(pre)}
   for p,h in before.items():
    if p.startswith('/refactor/')and not Path(p).resolve().is_relative_to(D.resolve()):assert references.get(p)==h,('Unpinned external dependency',p)
   run(prefix+['-MD','-MF',str(md),'-c',str(D/name),'-o',str(obj)],name+'-compile',90)
   assert deps(md)==set(before)and all(sha(p)==h for p,h in before.items())
   q['compiled'].append(dict(source=str(D/name),source_SHA256=sha(D/name),object=str(obj),object_SHA256=sha(obj),MD=str(md),MD_SHA256=sha(md),dependency_SHA256=before))
  exact=prior['localized_private_object']['only_localized'];source=Path(q['compiled'][0]['object']);localized=B/'private_moe.only_7_C_wrappers_local.o'
  before=symbols(run(['nm','-g','--defined-only',str(source)],'before-private-global'));assert len(exact)==7 and all(before.get(k)=='T'for k in exact)
  run(['objcopy',*['--localize-symbol='+name for name in exact],str(source),str(localized)],'localize-exact-seven-clone-wrappers')
  after=symbols(run(['nm','-g','--defined-only',str(localized)],'after-private-global'));assert after=={k:v for k,v in before.items()if k not in exact}
  sections=text_sections(source);assert sections==text_sections(localized)
  q['localized_private_object']=dict(original=str(source),object=str(localized),object_SHA256=sha(localized),only_localized=exact,all_text_sections_SHA256=sections,other_globals_and_all_text_bytes_unchanged=True)
  link=next(v['argv']for v in prior['commands']if v['label']=='isolated-link-reuse-old-GPU-33objects')[:]
  for previous,new in [(prior['compiled'][1]['object'],q['compiled'][1]['object']),(prior['localized_private_object']['object'],str(localized))]:assert link.count(previous)==1;link[link.index(previous)]=new
  oldgpu=c['old_GPU_object']['object'];assert link.count(oldgpu)==1;link.remove(oldgpu)
  lib=B/'libKimi_lowrank_TP_rowshard_r440.so';link[link.index('-o')+1]=str(lib);link=[('-Wl,-Map,'+str(B/'link.map'))if v.startswith('-Wl,-Map,')else v for v in link];assert '-Wl,-Bsymbolic'in link and '-Wl,--no-undefined'in link
  run(link,'isolated-full-native-plus-new-consumer-link-no432GPU',45)
  dyn=symbols(run(['nm','-D','--defined-only',str(lib)],'new-symbols'));old=symbols(run(['nm','-D','--defined-only',native['native']],'qualified8df-symbols'));strong={k:v for k,v in old.items()if v in ['T','D','B','R']};assert all(dyn.get(k)==v for k,v in strong.items()),'Old qualified strong/CABI removed'
  for name in ['modern_kimi_lowrank_full_moe_r440','modern_kimi_lowrank_full_moe_result_r440']:assert dyn.get(name)=='T'
  dynamic=run(['readelf','-d',str(lib)],'new-dynamic');assert 'SYMBOLIC'in dynamic
  allgather='_ZN8infinilm6engine11distributed10allgather_ERNS_7runtime7RuntimeENS2_6TensorERKS5_Pv';common=refs['actual_Common_extension'];assert sha(common['path'])==common['SHA256'];assert symbols(run(['nm','-D','--defined-only',common['path']],'actual-Common-AllGather-provider')).get(allgather)=='T'
  ldd=run(['ldd','-r',str(lib)],'new-ldd-r');assert 'not found'not in ldd and 'undefined symbol'not in ldd
  for name,expected in native['actual_dependency_library_SHA256'].items():
   rows=[v.split()for v in ldd.splitlines()if v.split()and v.split()[0]==name];assert len(rows)==1 and rows[0][1]=='=>'and sha(rows[0][2])==expected['SHA256']
  helper=refs['helper']['path'];assert sha(helper)==refs['helper']['SHA256'];rows=[v.strip()for v in ldd.splitlines()if helper in v];assert len(rows)==1 and rows[0].split()[0]==helper and helper in dynamic
  q.update(passed=True,completed=True,library=str(lib),library_SHA256=sha(lib),old_strong_symbols_preserved=len(strong),reused_original33=native['all_linked_object_SHA256'],old432GPU_linked=False,actual_AllGather_provider=common,all_dependencies_resolved=True,new_device_or_math_qualification=False,GPU_GO=False,Model_GO=False)
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 finally:
  q['elapsed_s']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['passed','actual_CPU','error','elapsed_s','library_SHA256']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
