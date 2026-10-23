import pathlib,json,hashlib,shutil,subprocess,shlex,concurrent.futures
r=pathlib.Path('/refactor');parent=r/'candidates/deep-combined-r7169';b=r/'candidates/deep-tailSUM-r7212';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();mp=r/'results/deep-combined-r7169-build.json';assert sha(mp)=='4dbb7e7a982cd73947db8a5ff65e2293692f8d26931fc241902888dc7be9768f';old=json.loads(mp.read_text());assert sha(old['full_library'])=='e8f0c19746e54ff2ababfbe1bb2a1704953f87a9bfc9262dcd35148c5d469201';assert sha(old['partial_library'])=='f526a3a53b107eac88b142e52b9b0bfb9d63336d3e8377539d2d974426b89e0c';spec=json.loads((r/'incoming/deep-tailSUM-source-manifest-r7211.json').read_text());assert not b.exists();b.mkdir();shutil.copytree(parent/'incoming',b/'incoming');hist={n:h for u in old['compiled_units']for n,h in u['deps'].items()};fresh={}
for dst,src in spec['input_source_map'].items():
 p=r/'incoming'/src;assert sha(p)==spec['source_SHA256'][src];t=b/'incoming'/dst;t.write_bytes(p.read_bytes());fresh[str(t)]=sha(t)
argv=old['full_link_argv'];oi=argv.index('-o');flags=['g++','-I'+str(b/'incoming')]+[v for i,v in enumerate(argv)if i not in[0,oi,oi+1]and not v.endswith('.o')and v!='-shared'and not v.startswith(('-l','-L','-Wl,'))];parse=lambda p:list(dict.fromkeys(shlex.split(p.read_text().replace('\\\n',' '))[1:]));sources=[b/'incoming'/n for n in spec['input_source_map']if n.endswith('.cpp')]
def compile_one(src):
 obj=b/(src.name+'.o');md=b/(src.name+'.d');fm=b/(src.name+'.M');p=subprocess.run(flags+['-M','-MF',str(fm),'-MT',str(obj),str(src)],capture_output=True,text=True,timeout=120);assert p.returncode==0,p.stderr;deps=parse(fm);before={n:sha(n)for n in deps};extra={}
 for n,h in before.items():
  original=n.replace(str(b)+'/',str(parent)+'/')
  if n in fresh:assert h==fresh[n]
  elif original in hist:assert h==hist[original]
  else:assert n.startswith(('/usr/include/','/usr/lib/gcc/')),('unexpected_header_shadow',n);extra[n]=h
 p=subprocess.run(flags+['-MD','-MF',str(md),'-c',str(src),'-o',str(obj)],capture_output=True,text=True,timeout=300);(b/(src.name+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr;assert set(parse(md))==set(deps)
 for n,h in before.items():assert sha(n)==h
 return {'source':str(src),'source_SHA256':sha(src),'object':str(obj),'object_SHA256':sha(obj),'deps':before,'new_standard_headers':extra,'MD_equals_fresh_M':True}
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:units=list(pool.map(compile_one,sources))
side=str(b/'incoming/deep_tail_prefill_scope_r7211.hpp');reduce=str(b/'incoming/deep_tail_SUM_reduce_r7211.hpp');assert all(side in u['deps']for u in units);assert [pathlib.Path(u['source']).name for u in units if reduce in u['deps']]==['deepseek_v4_moe_prefill_scatter_r194.cpp'];replacement={next(v['object']for v in old['compiled_units']if pathlib.Path(v['source']).name==pathlib.Path(u['source']).name):u['object']for u in units};libraries={};links={};exports={}
for which in ['full','partial']:
 args=old[which+'_link_argv'];link=[replacement.get(n,n)for n in args];lib=b/f'libdeep_tailSUM_{which}_r7212.so';link[link.index('-o')+1]=str(lib)
 for n,h in old[which+'_link_objects'].items():assert sha(n)==h
 for n in replacement:assert n not in link
 p=subprocess.run(link,capture_output=True,text=True,timeout=120);(b/f'link-{which}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr;nm=subprocess.check_output(['nm','-D','--defined-only',str(lib)],text=True);dynamic=subprocess.check_output(['readelf','-d',str(lib)],text=True);assert 'SYMBOLIC'in dynamic
 for n in ['deep_moe_tail_get_r7211','deep_moe_tail_set_r7211','deep_moe_tail_calls_r7211']:assert any(v.split()[-1]==n for v in nm.splitlines())
 libraries[which]={'path':str(lib),'SHA256':sha(lib)};links[which]={'argv':link,'objects':{n:sha(n)for n in link if n.endswith('.o')}};exports[which]={'nm':nm,'readelf':dynamic}
fixture=r/'incoming/deep_tail_metadata_fixture_r7212.cpp';exe=b/'metadata-fixture';fdep=b/'metadata-fixture.d';p=subprocess.run(flags+['-UNDEBUG','-MD','-MF',str(fdep),str(fixture),'-o',str(exe)],capture_output=True,text=True,timeout=120);(b/'metadata-fixture.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr;fdynamic=subprocess.check_output(['readelf','-d',str(exe)],text=True);assert not any(n in fdynamic for n in ['libinfini','libcuda','_infinilm']);p=subprocess.run([str(exe)],capture_output=True,text=True,timeout=10);assert p.returncode==0
binding=json.loads((r/'results/deep-SUM-r7204-build.json').read_text())['actual_DTK_bindings']
for n,v in binding.items():assert sha(n)==v['SHA256']
out={'completed':True,'scope':'CPU2TUcompile/link +pureCPUmetadatafixture only;noDSOdlopen/GPU/Model','parent_full_SHA256':sha(old['full_library']),'parent_partial_SHA256':sha(old['partial_library']),'parent_manifest_SHA256':sha(mp),'input_manifest_SHA256':sha(r/'incoming/deep-tailSUM-source-manifest-r7211.json'),'input_freeze':fresh,'compiled_units':units,'reverse_dependency':'sidecarinboth2TUs;newreduceonlyMoETU;oldcore/Layout/Frame/Counts headersunchanged','libraries':libraries,'link':links,'exports':exports,'actual_DTK_bindings':binding,'ABI':old['ABI'],'CPU_metadata_fixture':{'exit':p.returncode,'source_SHA256':sha(fixture),'executable_SHA256':sha(exe),'dynamic':fdynamic,'dependencies':{n:sha(n)for n in parse(fdep)},'scope':'productionmatches_metadatapredicate trueonlysingle512allpast0;512pastnon0/multi/T1/T16/offshape false;noGPUlibrarieslinked'}};p=r/'results/deep-tailSUM-r7212-build.json'
with p.open('x')as f:json.dump(out,f,indent=2)
print(json.dumps({'completed':True,'libraries':libraries,'build_SHA256':sha(p)}))
