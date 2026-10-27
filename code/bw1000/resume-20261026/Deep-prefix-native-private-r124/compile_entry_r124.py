"""Generate exact private HOST compile/link plan; no compilation by default.

Root must review then explicitly invoke --root-reviewed-build in the existing
CPU build environment. Never uploads, submits jobs, dlopens or executes a DSO.
"""
import argparse,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--root-reviewed-build',action='store_true');p.add_argument('--extra-include',action='append',default=[]);a=p.parse_args()
binding=json.loads((ROOT/'actual_binding_readonly_r124.json').read_text());link=binding['current_link'];objects=link['objects'];sources={}
for unit in binding['units']:sources[unit['object']]=Path(unit['source']).name
overrides={
 '/refactor/candidates/deep-attention-inner-r9193/deepseek_v4_swa_attention.cpp.o':'deepseek_v4_swa_attention.cpp',
 '/refactor/candidates/deep-attention-inner-r9193/deepseek_v4_hca_attention_batch_r784.cpp.o':'deepseek_v4_hca_attention_batch_r784.cpp',
 '/refactor/candidates/deep-attention-inner-r9193/deepseek_v4_csa_attention_batch_r784.cpp.o':'deepseek_v4_csa_attention_batch_r784.cpp',
 '/refactor/candidates/deep-owner-device-r8918/deepseek_v4_native_routed_decoder_r20.cpp.o':'deepseek_v4_native_routed_decoder_r20.cpp',
 '/refactor/candidates/deep-c4-smallt-r9252/deep_c4_smallt_r9252.cpp.o':'deep_c4_smallt_r9252.cpp'}
sources.update(overrides);assert len(sources)==9 and all(x in objects for x in sources)
argv=link['argv'];compiler=argv[0];flags=[];j=1
while j<len(argv):
 x=argv[j]
 if x=='-o':j+=2;continue
 if x.endswith('.o') or x=='-shared' or x.startswith(('-L','-l','-Wl,')):j+=1;continue
 flags.append(x);j+=1
extra=['/refactor/candidates/deep-owner-device-r8918/src','/refactor/incoming/Deep-attention-inner-r9193','/refactor/candidates/deep-tailSUM-r7212/incoming','/refactor/candidates/deep-c4-smallt-r9252','/refactor/candidates/deep-owner-events-r8900/src']+a.extra_include
flags=['-I'+str(ROOT/'src'),'-I'+str(ROOT)]+['-I'+x for x in extra]+flags
out=ROOT/'build';units=[]
for old,name in sources.items():
 units.append(dict(replaces=old,replaces_SHA256=objects[old],source=str(ROOT/'src'/name),source_SHA256=sha(ROOT/'src'/name),object=str(out/(name+'.o')),argv=[compiler]+flags+['-MD','-MF',str(out/(name+'.d')),'-c',str(ROOT/'src'/name),'-o',str(out/(name+'.o'))]))
name='deep_prefix_native_r124.cpp';units.append(dict(replaces=None,source=str(ROOT/name),source_SHA256=sha(ROOT/name),object=str(out/(name+'.o')),argv=[compiler]+flags+['-MD','-MF',str(out/(name+'.d')),'-c',str(ROOT/name),'-o',str(out/(name+'.o'))]))
replace={u['replaces']:u['object'] for u in units if u['replaces']};newlink=[replace.get(x,x) for x in argv];newlink[newlink.index('-o')+1]=str(out/'libdeep_prefix_private_r124.so');newlink.insert(1,units[-1]['object'])
plan=dict(status='UNCOMPILED_HOST_SOURCE_PLAN',baseline_DSO_SHA256='ac9370c8f188bfda8555718c52690b914883fc6b02b9267cbd279be3ecdd3835',baseline_objects=len(objects),rebuild_existing_HOST_TUs=9,new_HOST_TUs=1,unchanged_objects=len(objects)-9,units=units,link_argv=newlink,
 requirements=['All9 replacement consumers MUST rebuild, including NativeSparseModel allocator and TemporalCompressor cache allocator. Never reuse old layout objects.',
 'Compiler dependency MD must resolve all modified Deep class headers to private src, and transaction/native adapter headers to this private root. Reject duplicate class definitions or old altered headers.',
 'Extra include directories are retained known ac937 source contexts. Resolve missing custom HOST headers using original per-object CPU compile receipts; do not substitute a different runtime/SDK/math header.',
 'This command builds HOST objects only after explicit root review. Linking and CPU contract test execution remain separate commands for root. No dlopen, weights, GPU or native performance qualification.'])
(ROOT/'compile_plan_r124.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
if a.root_reviewed_build:
 out.mkdir(exist_ok=True)
 for old in objects:assert sha(old)==objects[old],old
 for u in units:
  assert sha(u['source'])==u['source_SHA256'];r=subprocess.run(u['argv'],text=True,capture_output=True);(out/(Path(u['source']).name+'.log')).write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr[-6000:]
  # C++ depfile paths are checked before a future link can be accepted.
  dep=(out/(Path(u['source']).name+'.d')).read_text();forbidden=[]
  for token in dep.replace('\\\n',' ').split():
   name=Path(token).name
   if name in {f.name for f in (ROOT/'src').glob('*.hpp')} and str(ROOT/'src') not in token:forbidden.append(token)
  assert not forbidden,forbidden
 plan['status']='HOST_OBJECTS_COMPILED_NO_LINK_NO_EXECUTION';plan['compiled']=[dict(object=u['object'],SHA256=sha(u['object'])) for u in units];(ROOT/'compile_receipt_r124.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(json.dumps(dict(status=plan['status'],rebuild=9,new=1,unchanged=len(objects)-9,plan=str(ROOT/'compile_plan_r124.json'))))
