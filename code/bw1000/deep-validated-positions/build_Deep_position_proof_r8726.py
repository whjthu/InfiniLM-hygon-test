import pathlib,json,hashlib,subprocess,shlex,time
r=pathlib.Path('/refactor');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();began=time.monotonic();mp=r/'results/deep-tailSUM-r7212-build.json';assert sha(mp)=='fc7065a9b56255bb710dbedb53a1b71e6c017f7bdc23bc2a86a41664789a3b0b';parent=json.loads(mp.read_text());old=json.loads((r/'results/deep-combined-r7169-build.json').read_text());cp=r/'candidates/deep-C128-weight-cache-r8684/CPU-build.json';assert sha(cp)=='e3d4d7adc0aa9d67b460f0f6cdb6aff85b8b0d4cb053820af0f5c20fe619e28b';cache=json.loads(cp.read_text());native=next(u for u in parent['compiled_units']if pathlib.Path(u['source']).name=='deepseek_v4_native_routed_decoder_r20.cpp');spec=json.loads((r/'incoming/Deep-position-proof-source-r8726.json').read_text());b=r/'candidates/deep-position-proof-r8726';assert not b.exists();b.mkdir();(b/'incoming').mkdir()
for n,h in spec['files'].items():
 p=r/'incoming'/n;assert sha(p)==h;(b/'incoming'/n).write_bytes(p.read_bytes())
# Compile the actual shared header against a tiny CPU mock only for scope/key contract tests.
mock=b/'mock/runtime';mock.mkdir(parents=True);(mock/'runtime.hpp').write_bytes((b/'incoming/Deep_position_mock_runtime_r8726.hpp').read_bytes());exe=b/'CPU-contract';cmd=['g++','-std=c++17','-O0','-UNDEBUG','-I'+str(b/'mock'),'-I'+str(b/'incoming'),str(b/'incoming/Deep_position_CPU_contract_r8726.cpp'),'-o',str(exe)];p=subprocess.run(cmd,capture_output=True,text=True,timeout=30);assert p.returncode==0,p.stderr;p=subprocess.run([str(exe)],capture_output=True,text=True,timeout=10);assert p.returncode==0,p.stderr;contract=json.loads(p.stdout);assert contract['passed'];print(json.dumps({'CPU_contract':contract}),flush=True)
argv=old['full_link_argv'];oi=argv.index('-o');nf=['g++','-I'+str(b/'incoming'),'-I'+str(r/'candidates/deep-tailSUM-r7212/incoming')]+[v for i,v in enumerate(argv)if i not in[0,oi,oi+1]and not v.endswith('.o')and v!='-shared'and not v.startswith(('-l','-L','-Wl,'))];cf=['g++','-I'+str(b/'incoming')]+cache['compile_argv'][1:];cf=[v for v in cf if not v.startswith(('-l','-L','-Wl,'))];parse=lambda p:list(dict.fromkeys(shlex.split(p.read_text().replace('\\\n',' '))[1:]));compobj='/refactor/candidates/deep-c4-projection-model-r5920/objects/unit-20.o';assert sha(compobj)=='23463d032e0517575110e200fac2a8041b038f434564bab41ea788b3e4e5da64';assert sha(native['source'])==spec['baseline_native_SHA256'];assert sha(native['object'])==native['object_SHA256'];assert sha('/refactor/candidates/deep-c4-projection-model-r5920/incoming/deepseek_v4_compressor_fp32_c128_r19.cpp')==spec['baseline_compressor_SHA256']
# Fresh dependency receipts are checked against actual parent receipts wherever present;
# unchanged headers first seen by this added include remain explicitly hashed in MD.
history={n:h for u in parent['compiled_units']for n,h in u['deps'].items()};extra=cache['actual_MD'];
if isinstance(extra,dict):
 for n,h in extra.items():
  if isinstance(h,str)and len(h)==64:history[n]=h
for u in old['compiled_units']:history.update(u['deps'])
units=[];replace={}
for name,flags,oldobj in [('deep_native_position_proof_r8726.cpp',nf,native['object']),('deep_compressor_position_proof_r8726.cpp',cf,compobj)]:
 src=b/'incoming'/name;obj=b/(src.stem+'.o');fm=obj.with_suffix('.M');md=obj.with_suffix('.d');p=subprocess.run(flags+['-M','-MF',str(fm),'-MT',str(obj),str(src)],capture_output=True,text=True,timeout=60);assert p.returncode==0,p.stderr;deps=parse(fm);before={n:sha(n)for n in deps};new={}
 for n,h in before.items():
  if n in history:assert h==history[n],('parent_header_changed',n)
  elif n.startswith(str(b/'incoming')+'/'):assert h==spec['files'][pathlib.Path(n).name]
  else:new[n]=h
 p=subprocess.run(flags+['-MD','-MF',str(md),'-c',str(src),'-o',str(obj)],capture_output=True,text=True,timeout=120);(b/(src.stem+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr;assert set(parse(md))==set(deps)
 for n,h in before.items():assert sha(n)==h
 units.append(dict(source=str(src),source_SHA256=sha(src),object=str(obj),object_SHA256=sha(obj),flags=flags,deps=before,new_dependency_receipts=new,MD_equals_fresh_M=True));replace[oldobj]=str(obj)
libs={};links={}
for which in ['full','partial']:
 for n,h in parent['link'][which]['objects'].items():assert sha(n)==h
 args=[replace.get(n,n)for n in parent['link'][which]['argv']];lib=b/f'libdeep_position_proof_{which}_r8726.so';args[args.index('-o')+1]=str(lib)
 for n in replace:assert n not in args
 p=subprocess.run(args,capture_output=True,text=True,timeout=60);(b/f'link-{which}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
 nm=subprocess.check_output(['nm','-D','--defined-only',str(lib)],text=True);dynamic=subprocess.check_output(['readelf','-d',str(lib)],text=True);assert 'SYMBOLIC'in dynamic;names=[v.split()[-1]for v in nm.splitlines()];assert names.count('deep_position_proof_counts_r8726')==1
 for n in ['deep_moe_tail_set_r7211','deep_output_tile_calls_r4808','deep_c4_policy_r5920']:assert n in names or n=='deep_c4_policy_r5920'
 (b/f'{which}.nm').write_text(nm);libs[which]={'path':str(lib),'SHA256':sha(lib)};links[which]={'argv':args,'objects':{n:sha(n)for n in args if n.endswith('.o')}}
for n,v in parent['actual_DTK_bindings'].items():assert sha(n)==v['SHA256']
q=dict(completed=True,scope='CPU2TU/privateheader compile+link and mockCPU scope contracts only; NO dlopen/GPU/weights',parent_manifest_SHA256=sha(mp),parent_libraries=parent['libraries'],source_spec_SHA256=sha(r/'incoming/Deep-position-proof-source-r8726.json'),compiled_units=units,libraries=libs,link=links,actual_DTK_bindings=parent['actual_DTK_bindings'],ABI=parent['ABI'],CPU_contract={'actual_header':True,'runtime_mock_only':True,**contract},CPU_seconds=time.monotonic()-began,default_OFF=True,class_layout_changed=False,math_changed=False,feature_GPU_NOT_RUN=True);out=r/'results/deep-position-proof-r8726-build.json'
with out.open('x')as f:json.dump(q,f,indent=2)
print(json.dumps({'completed':True,'build_SHA256':sha(out),'libraries':libs,'CPU_seconds':q['CPU_seconds']}))
