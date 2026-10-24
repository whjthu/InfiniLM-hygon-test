import pathlib,json,hashlib,subprocess,shlex,time
r=pathlib.Path('/refactor');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();parentp=r/'results/deep-tailSUM-r7212-build.json';assert sha(parentp)=='fc7065a9b56255bb710dbedb53a1b71e6c017f7bdc23bc2a86a41664789a3b0b';parent=json.loads(parentp.read_text());old=json.loads((r/'results/deep-combined-r7169-build.json').read_text());unit=next(u for u in old['compiled_units']if pathlib.Path(u['source']).name=='deepseek_v4_attention_output.cpp');assert sha(unit['source'])==unit['source_SHA256']=='31e80fb1109308809a030896681ceacc17fbe5b76f1a61b528389456f683f960';assert sha(unit['object'])==unit['object_SHA256']=='b6d87d042a48665fee3310249506a3b1f6768001fc3b564f0a4fda69afa4dd99'
for which in ['full','partial']:
 assert parent['link'][which]['argv'].count(unit['object'])==1
 assert sha(parent['libraries'][which]['path'])==parent['libraries'][which]['SHA256']
 for n,h in parent['link'][which]['objects'].items():assert sha(n)==h
for n,h in unit['deps'].items():assert sha(n)==h,('parent_dependency_changed',n)
b=r/'candidates/deep-output-clock-r8713';assert not b.exists();b.mkdir();source=r/'incoming/deep_attention_output_clock_r8713.cpp';spec=json.loads((r/'incoming/Deep-output-clock-source-r8713.json').read_text());assert sha(source)==spec['new_source_SHA256'];target=b/'deepseek_v4_attention_output.cpp';target.write_bytes(source.read_bytes());argv=old['full_link_argv'];oi=argv.index('-o');flags=['g++','-I'+str(r/'candidates/deep-combined-r7169/incoming')]+[v for i,v in enumerate(argv)if i not in [0,oi,oi+1]and not v.endswith('.o')and v!='-shared'and not v.startswith(('-l','-L','-Wl,'))];obj=b/'attention-output.o';fm=b/'attention-output.M';md=b/'attention-output.d';parse=lambda p:list(dict.fromkeys(shlex.split(p.read_text().replace('\\\n',' '))[1:]));began=time.monotonic();cmd=flags+['-M','-MF',str(fm),'-MT',str(obj),str(target)];p=subprocess.run(cmd,capture_output=True,text=True,timeout=120);assert p.returncode==0,p.stderr;deps=parse(fm);before={n:sha(n)for n in deps};newstd={}
for n,h in before.items():
 if n==str(target):assert h==spec['new_source_SHA256']
 elif n in unit['deps']:assert h==unit['deps'][n]
 else:assert n.startswith(('/usr/include/','/usr/lib/gcc/')),('unexpected_dependency',n);newstd[n]=h
p=subprocess.run(flags+['-MD','-MF',str(md),'-c',str(target),'-o',str(obj)],capture_output=True,text=True,timeout=300);(b/'compile.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr;assert set(parse(md))==set(deps)
for n,h in before.items():assert sha(n)==h
links={};libs={}
for which in ['full','partial']:
 argv=parent['link'][which]['argv'];args=[str(obj)if n==unit['object']else n for n in argv];lib=b/f'libdeep_output_clock_{which}_r8713.so';args[args.index('-o')+1]=str(lib);assert unit['object']not in args;assert args.count(str(obj))==1
 p=subprocess.run(args,capture_output=True,text=True,timeout=120);(b/f'link-{which}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
 nm=subprocess.check_output(['nm','-D','--defined-only',str(lib)],text=True);dyn=subprocess.check_output(['readelf','-d',str(lib)],text=True);assert 'SYMBOLIC'in dyn;names={v.split()[-1]for v in nm.splitlines()};assert 'deep_output_clock_config_r8713'in names
 for n in ['deep_output_tile_calls_r4808','deep_moe_tail_set_r7211','deep_hc_canonical_policy_r7130','modern_deep_layout_policy_r7052']:assert n in names
 (b/f'{which}.nm').write_text(nm);(b/f'{which}.dynamic').write_text(dyn);libs[which]={'path':str(lib),'SHA256':sha(lib)};links[which]={'argv':args,'objects':{n:sha(n)for n in args if n.endswith('.o')}}
for n,v in parent['actual_DTK_bindings'].items():assert sha(n)==v['SHA256']
q={'completed':True,'scope':'Private1TUCPUcompile+2links only, noCDLL/dlopen/GPU/weightload','libraries':libs,'compiled_unit':{'source':str(target),'source_SHA256':sha(target),'object':str(obj),'object_SHA256':sha(obj),'flags':flags,'deps':before,'MD_equals_fresh_M':True,'new_standard_headers':newstd},'replaced_parent_unit':unit['object'],'replaced_parent_object_SHA256':unit['object_SHA256'],'parent_manifest_SHA256':sha(parentp),'parent_libraries':parent['libraries'],'link':links,'actual_DTK_bindings':parent['actual_DTK_bindings'],'ABI':parent['ABI'],'CPU_seconds':time.monotonic()-began,'production_feature_default_OFF':True,'new_C_API':'deep_output_clock_config_r8713 only diagnostic counters','performance_adopted':False};out=r/'results/deep-output-clock-r8713-build.json'
with out.open('x')as f:json.dump(q,f,indent=2)
print(json.dumps({'completed':True,'build_SHA256':sha(out),'object_SHA256':sha(obj),'libraries':libs,'CPU_seconds':q['CPU_seconds']}))
