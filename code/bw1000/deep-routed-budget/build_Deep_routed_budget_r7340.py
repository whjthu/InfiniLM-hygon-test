"""One adapter TU, actual Deep7212 SDK/link chain, no GPU/dlopen."""
import pathlib,json,hashlib,subprocess,shlex,shutil,os,time
r=pathlib.Path('/refactor');base=r/'candidates/deep-routed-budget-r7340';assert not base.exists();(base/'src').mkdir(parents=True)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();mp=r/'incoming/Deep-routed-budget-CPU-manifest-r7340.json';m=json.loads(mp.read_text());parent=json.loads((r/'results/deep-tailSUM-r7212-build.json').read_text());libparent=pathlib.Path(parent['libraries']['full']['path']);assert sha(libparent)==m['parent_DSO_SHA256']
for n,h in m['sources_SHA256'].items():assert sha(r/'incoming'/n)==h;shutil.copyfile(r/'incoming'/n,base/'src'/n)
for n,v in parent['actual_DTK_bindings'].items():assert sha(n)==v['SHA256']
argv=parent['link']['full']['argv'];oi=argv.index('-o');flags=[v for j,v in enumerate(argv)if j not in[0,oi,oi+1]and not v.endswith('.o')and v!='-shared'and not v.startswith(('-l','-L','-Wl,'))and v!='-I/refactor/incoming']
flags=['-I'+str(base/'src'),'-I/refactor/candidates/deep-tailSUM-r7212/incoming','-I/opt/dtk/cuda/cuda/include',*flags];compiler=shutil.which('g++');commands=[];report={'completed':False,'CPU_only':True,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']};start=time.perf_counter()
def run(a,label):
 commands.append(a);p=subprocess.run(a,text=True,capture_output=True);(base/(label+'.log')).write_text(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError((label,p.returncode,p.stderr[-7000:]))
 return p.stdout
try:
 src=base/'src/deep_routed_budget_adapter_r7338.cpp';obj=base/'adapter.o';md=base/'adapter.d'
 run([compiler,*flags,'-MD','-MF',str(md),'-c',str(src),'-o',str(obj)],'compile')
 parse=lambda s:set(shlex.split(s.replace('\\\n',' ').split(':',1)[1]));deps=parse(md.read_text());assert deps==parse(run([compiler,*flags,'-M',str(src)],'fresh-M'))
 header='/refactor/candidates/deep-tailSUM-r7212/incoming/deepseek_routed_mlp_batch_r20.hpp';assert header in deps and sha(header)==m['original_header_SHA256'];assert '/opt/dtk/cuda/cuda/include/cuda_runtime_api.h'in deps
 link=[compiler,str(obj),str(libparent),'-shared','-fPIC'];link +=[v for v in argv if v.startswith(('-L','-l','-Wl,'))and v!='-Wl,-Bsymbolic'];lib=base/'libdeep_routed_budget_r7340.so';link+=['-Wl,--no-undefined','-Wl,-rpath,'+str(libparent.parent),'-o',str(lib)];run(link,'link')
 exports=run(['nm','-D','--defined-only',str(lib)],'exports');undefined=run(['nm','-D','--undefined-only',str(lib)],'undefined');dynamic=run(['readelf','-d',str(lib)],'dynamic');assert str(libparent)in dynamic
 for n in ['create','forward','report','error','destroy']:assert any(l.split()[-1]=='deep_routed_budget_'+n+'_r7338'for l in exports.splitlines())
 report.update(completed=True,elapsed_seconds=time.perf_counter()-start,source_manifest_SHA256=sha(mp),source_SHA256={n:sha(base/'src'/n)for n in m['sources_SHA256']},object_SHA256=sha(obj),MD_SHA256=sha(md),fresh_M_equal_MD=True,dependencies_SHA256={p:sha(p)for p in deps},library=str(lib),library_SHA256=sha(lib),parent_DSO_SHA256=sha(libparent),DTK_bindings=parent['actual_DTK_bindings'],commands=commands,exports=exports,undefined=undefined,dynamic=dynamic,NOT_RUN=['CPUdlopen','GPU','weights','Model'])
except BaseException as e:
 report.update(error=repr(e),commands=commands);(base/'CPU-failure.json').write_text(json.dumps(report,indent=2));raise
else:
 (base/'CPU-build.json').write_text(json.dumps(report,indent=2));(r/'results/deep-routed-budget-CPU-r7340.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k]for k in['completed','producer','elapsed_seconds','library_SHA256']}))
