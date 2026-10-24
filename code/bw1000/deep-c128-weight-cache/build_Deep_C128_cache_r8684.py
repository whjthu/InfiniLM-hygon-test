"""One original compressor TU -> current0e051 full/partial private links; CPU only."""
import pathlib,json,hashlib,subprocess,shlex,time
r=pathlib.Path('/refactor');b=r/'candidates/deep-C128-weight-cache-r8684';assert not b.exists();b.mkdir()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
p=r/'results/deep-c4-projection-model-r5920-build.json';assert sha(p)=='9fd84fb279396985f8ad2fe56a4f081adeccad823e95ca1dd881bfdd82706c70';parent=json.loads(p.read_text());base=r/'candidates/deep-c4-projection-model-r5920'
p=r/'results/deep-tailSUM-r7212-build.json';assert sha(p)=='fc7065a9b56255bb710dbedb53a1b71e6c017f7bdc23bc2a86a41664789a3b0b';current=json.loads(p.read_text());assert sha(current['libraries']['full']['path'])==current['libraries']['full']['SHA256']=='0e0512551483457af01944e9c9298442f499d959836a1c5b66dfa438a3c6c4ea'
u=next(u for u in parent['units']if pathlib.Path(u['source']).name=='deepseek_v4_compressor_fp32_c128_r19.cpp');assert sha(u['source'])==u['source_sha256']=='8e3160a8d3defff085c888dea3d78bf1d09f0b8d3c4d6cbf2d8f85735041feee';assert sha(u['object'])==u['object_sha256']=='23463d032e0517575110e200fac2a8041b038f434564bab41ea788b3e4e5da64'
src=r/'incoming/deep_C128_weight_cache_r8681.cpp';assert sha(src)=='75b1b91deef6fd823574c7d121da63ffa4f4d0d5c0d51bdc3048b47ebee4d7d7';private=b/src.name;private.write_bytes(src.read_bytes())
env={'BASE':str(base),'COMMON':str(r/'candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM'),'LM':str(r/'candidates/rebased-5f91ad7-r3668/Model/Deep/InfiniLM'),'SDK':str(r/'install/hygon-deep-5f91ad7-r3672'),'HELPER':str(r/'candidates/models-5f91ad7-r3675/Deep/helpers'),'ABI':str(parent['ABI'])}
argv=shlex.split((base/'compile.sh').read_text().replace('\\\n',' '))
for i,x in enumerate(argv):
 for k,v in env.items():x=x.replace('${'+k+'}',v).replace('$'+k,v)
 argv[i]=x
oi=argv.index('-o');flags=[x for i,x in enumerate(argv)if i not in[oi,oi+1]and not x.endswith(('.cpp','.o'))and x!='-shared']
obj=b/'compressor-cache.o';md=b/'compressor-cache.d';fm=b/'compressor-cache.M';parse=lambda p:list(dict.fromkeys(shlex.split(p.read_text().replace('\\\n',' '))[1:]))
start=time.monotonic();z=subprocess.run(flags+['-M','-MF',str(fm),'-MT',str(obj),str(private)],capture_output=True,text=True,timeout=120);(b/'M.log').write_text(z.stdout+z.stderr);assert z.returncode==0,z.stderr
deps={n:sha(n)for n in parse(fm)};known={v['path']:v['sha256']for v in u['dependencies']}
for n,h in deps.items():
 if n==str(private):continue
 if n in known:assert known[n]==h,n
 else:assert n.startswith(('/usr/include/','/usr/lib/gcc/')),('unexpected_dependency',n)
header=next(n for n in deps if pathlib.Path(n).name=='deepseek_v4_temporal_compressor.hpp');assert deps[header]=='1b5de31aa0b27e28bb96f41cf68ee52ee58a4784e5ea3bf5e347b0c6e22b157a'
z=subprocess.run(flags+['-MD','-MF',str(md),'-c',str(private),'-o',str(obj)],capture_output=True,text=True,timeout=240);(b/'compile.log').write_text(z.stdout+z.stderr);assert z.returncode==0,z.stderr;assert set(parse(md))==set(deps)
for n,h in deps.items():assert sha(n)==h,n
links={}
for which in ['full','partial']:
 old=current['link'][which];assert old['argv'].count(u['object'])==1
 for n,h in old['objects'].items():assert sha(n)==h,n
 lib=b/f'libdeep_C128_weight_cache_{which}_r8684.so';args=[str(obj)if n==u['object']else n for n in old['argv']];args[args.index('-o')+1]=str(lib)
 z=subprocess.run(args,capture_output=True,text=True,timeout=120);(b/f'link-{which}.log').write_text(z.stdout+z.stderr);assert z.returncode==0,z.stderr
 nm=subprocess.check_output(['nm','-D','--defined-only',str(lib)],text=True);links[which]=dict(path=str(lib),SHA256=sha(lib),argv=args,objects={n:sha(n)for n in args if n.endswith('.o')},exports=nm)
for name in ['modern_native_engine_create','modern_native_engine_forward','modern_native_engine_destroy']:assert name in links['full']['exports']
out=dict(completed=True,CPU_only=True,GPU=False,dlopen=False,parent_DSO_SHA256=current['libraries']['full']['SHA256'],source_SHA256=sha(private),object_SHA256=sha(obj),ABI=parent['ABI'],header_SHA256=deps[header],compile_argv=flags,actual_MD=deps,MD_equals_fresh_M=True,link=links,seconds=time.monotonic()-start,scope='Oneunit20 replaced; classheader/layout unchanged; noGPU/Model/weights/runtimeinit; defaultOFF cacheflag loadtimeonly. Componentfixture and numerical gates NOT_RUN.')
with(b/'CPU-build.json').open('x')as f:json.dump(out,f,indent=2)
print(json.dumps(dict(completed=True,seconds=out['seconds'],receipt_SHA256=sha(b/'CPU-build.json'),libraries={k:{a:v[a]for a in ['path','SHA256']}for k,v in links.items()})))
