"""Known769 physical single-row span contract; two necessary HOST TUs + pure address contract, no SDK execution."""
from pathlib import Path
import hashlib,json,os,shlex,struct,subprocess,time,traceback
D=Path(__file__).resolve().parent;P=D.parent;B=Path('/refactor/candidates/Kimi-full-MoE-lowrank-M1-r437-CPU-row-span-v7');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
q=dict(passed=False,completed=False,GPU=0,Native_dlopen=False,Runtime_created=False,weights=0,new_production_HOST_TUs=2,pure_sched_contract_HOST_TUs=1,new_GPU_TUs=0,old_successful_TUs_recompiled=False,commands=[])
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
  m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert sha(D/'SOURCE_MANIFEST.json')==os.environ['KIMI_R437_ROWSPAN_V7_SHA']
  for p,h in m['files_SHA256'].items():assert sha(D/p)==h,p
  q.update(actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],entry_affinity=sorted(os.sched_getaffinity(0)),source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'));assert len(q['entry_affinity'])>=1
  oldpaths={'883':('/refactor/candidates/Kimi-full-MoE-lowrank-M1-r437-CPU/CPU-build.json','dfa142029c9fc168cddc3a7cffa133bbbad68a9ebdd8149f433402968680cb65'),'885':('/refactor/candidates/Kimi-full-MoE-lowrank-M1-r437-CPU-link-v2/CPU-build.json','8d664b80d9f4d069cfc19200af3da4a8d81205c03dfeaa4ce084c6bef663b84b'),'887':('/refactor/candidates/Kimi-full-MoE-lowrank-M1-r437-CPU-qualified-v3/CPU-build.json','00d74be5558d141d87e7cc598e7eaa10ef60e0d8ade1bb6f9e5387940c6571c2')};receipts={}
  for n,(p,h)in oldpaths.items():assert sha(p)==h;receipts[n]=json.loads(Path(p).read_bytes())
  old=receipts['883'];qualified=receipts['887'];assert qualified['passed']and sha(qualified['library'])==qualified['library_SHA256']
  references={}
  for v in old['compiled']:
   assert sha(v['object'])==v['object_SHA256']and sha(v['MD'])==v['MD_SHA256']
   for p,h in v['dependency_SHA256'].items():assert sha(p)==h;references[p]=h
  localized=qualified['localized_private_object'];assert sha(localized['object'])==localized['object_SHA256']
  c=json.loads((P/'BUILD_CONTRACT.json').read_bytes());native=c['parent_native'];assert sha(native['path'])==native['SHA256']and sha(native['native'])==native['native_SHA256']
  for p,h in native['all_linked_object_SHA256'].items():assert sha(p)==h
  g=c['old_GPU_object'];assert sha(g['object'])==g['object_SHA256']and sha(g['MD'])==g['MD_SHA256']
  for p,h in g['dependency_SHA256'].items():assert sha(p)==h
  assert sha(P/'cpu-first-submit-v5/component.cpp')==json.loads((D/'DERIVATION.json').read_bytes())['component_parent_SHA256']
  prior894=Path('/refactor/candidates/Kimi-full-MoE-lowrank-M1-r437-CPU-owner-v4/CPU-build.json');assert sha(prior894)=='0571e534902603c4ebb9dd11a15c198564388ff015b75910d73c1e76c090fdea';p894=json.loads(prior894.read_bytes());assert p894['passed']
  for p,h in p894['compiled']['dependency_SHA256'].items():assert sha(p)==h;references[p]=h
  prior898=Path('/refactor/candidates/Kimi-full-MoE-lowrank-M1-r437-CPU-first-submit-v5/CPU-build.json');assert sha(prior898)=='28ffd1dffa82f93b99919161321679809305c103728bb07eb1996cce0e270e8e';p898=json.loads(prior898.read_bytes());assert p898['passed']
  for p,h in p898['compiled']['dependency_SHA256'].items():assert sha(p)==h;references[p]=h
  q['actual_source_contract']=json.loads(run([__import__('sys').executable,'-B',str(D/'check_source_contract.py')],'actual-row-span-lease-and-source-contract',15))
  exe=B/'check_row_contract';run(['g++','-std=c++17','-O2','-pthread','-I'+str(D),str(D/'check_row_contract.cpp'),'-o',str(exe)],'pure-address-contract-compile',20)
  q['actual_pure_address_contract']=json.loads(run([str(exe)],'pure-address-and-rejection-boundaries',15));assert q['actual_pure_address_contract']['passed']
  q['compiled']=[]
  for name,base in [('private_moe.cpp',next(v['argv']for v in old['commands']if v['label']=='private_moe.cpp-compile')),('component.cpp',next(v['argv']for v in p898['commands']if v['label']=='changed-consumer-only-compile'))]:
   prefix=base[:base.index('-MD')];prefix.insert(1,'-I'+str(D));obj=B/(name+'.o');md=B/(name+'.d');run(prefix+['-MD','-MF',str(md),'-c',str(D/name),'-o',str(obj)],name+'-compile',90)
   deps=set(shlex.split(md.read_text().replace('\\\n',' ').split(':',1)[1]));pins={p:sha(p)for p in deps}
   for p,h in pins.items():
    if p.startswith('/refactor/')and not Path(p).resolve().is_relative_to(D.resolve()):assert references.get(p)==h,('New unpinned nonprivate dependency',p)
   q['compiled'].append(dict(source=str(D/name),source_SHA256=sha(D/name),object=str(obj),object_SHA256=sha(obj),MD=str(md),MD_SHA256=sha(md),dependency_SHA256=pins))
  localized_old=qualified['localized_private_object'];source=Path(q['compiled'][0]['object']);localized=B/'private_moe.only_7_C_wrappers_local.o'
  exact=localized_old['only_localized'];before=symbols(run(['nm','-g','--defined-only',str(source)],'before-new-private-global'));assert len(exact)==7 and all(before.get(k)=='T'for k in exact)
  run(['objcopy',*['--localize-symbol='+name for name in exact],str(source),str(localized)],'localize-exact-seven-clone-wrappers')
  after=symbols(run(['nm','-g','--defined-only',str(localized)],'after-new-private-global'));assert after=={k:v for k,v in before.items()if k not in exact}
  sections=text_sections(source);assert sections==text_sections(localized)
  q['localized_private_object']=dict(original=str(source),object=str(localized),object_SHA256=sha(localized),only_localized=exact,all_text_sections_SHA256=sections,other_globals_and_all_text_bytes_unchanged=True)
  link=next(v['argv']for v in p898['commands']if v['label']=='isolated-link-reuse-old-clone-GPU-33objects')[:];assert link.count(p898['compiled']['object'])==1 and link.count(localized_old['object'])==1
  link[link.index(p898['compiled']['object'])]=q['compiled'][1]['object'];link[link.index(localized_old['object'])]=str(localized)
  lib=B/'libKimi_full_MoE_lowrank_M1_r437_row_span_v7.so';link[link.index('-o')+1]=str(lib);link=[('-Wl,-Map,'+str(B/'link.map'))if v.startswith('-Wl,-Map,')else v for v in link];run(link,'isolated-link-reuse-old-GPU-33objects',45)
  dyn=symbols(run(['nm','-D','--defined-only',str(lib)],'new-symbols'));prior=symbols(run(['nm','-D','--defined-only',qualified['library']],'prior-symbols'));strong={k:v for k,v in prior.items()if v in ['T','D','B','R']};assert all(dyn.get(k)==v for k,v in strong.items()),'Old strong/CABI removed'
  assert dyn.get('modern_kimi_owner_affinity_preflight_r437')=='T';dynamic=run(['readelf','-d',str(lib)],'new-dynamic');assert 'SYMBOLIC'in dynamic
  ldd=run(['ldd','-r',str(lib)],'new-ldd-r');assert 'not found'not in ldd and 'undefined symbol'not in ldd
  for name,expected in native['actual_dependency_library_SHA256'].items():
   rows=[v.split()for v in ldd.splitlines()if v.split()and v.split()[0]==name];assert len(rows)==1 and rows[0][1]=='=>'and sha(rows[0][2])==expected['SHA256']
  helper=qualified['helper_binding']['path'];assert qualified['helper_SHA256']==sha(helper);rows=[v.strip()for v in ldd.splitlines()if helper in v];assert len(rows)==1 and rows[0].split()[0]==helper and '(NEEDED)'in dynamic and helper in dynamic
  q.update(passed=True,completed=True,library=str(lib),library_SHA256=sha(lib),old_strong_symbols_preserved=len(strong),old887_reused=oldpaths['887'],old_GPU757_FAIL_preserved=True,new_device_or_math_qualification=False,new_private_object=q['localized_private_object'],reused_original33=native['all_linked_object_SHA256'],reused_GPU=g,GPU_GO=False,Model_GO=False)
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 finally:
  q['elapsed_s']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['passed','actual_CPU','error','elapsed_s','library_SHA256']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
