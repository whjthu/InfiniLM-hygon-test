import hashlib,json,os,pathlib,subprocess,time,traceback
D=pathlib.Path(__file__).resolve().parent;BASE=D.parent;V2=BASE/'v2'
OUT=pathlib.Path('/refactor/candidates/Deep-PD-native-r382-cpu-v3')
O1=pathlib.Path('/refactor/candidates/Deep-PD-native-r382-cpu-v1');O2=pathlib.Path('/refactor/candidates/Deep-PD-native-r382-cpu-v2')
C=json.loads((BASE/'BUILD_CONTRACT.json').read_bytes());R1=json.loads((V2/'REUSE_CPU765.json').read_bytes());R2=json.loads((D/'REUSE_CPU766.json').read_bytes())
START=time.monotonic();result={'completed':False,'passed':False,'CPU_only':True,'GPU_initialized':False,'Native_dlopen':False,'Model_loaded':False,'GPU_TUs_compiled':0,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def run(a,label,timeout=240):
    z=subprocess.run([str(x)for x in a],capture_output=True,text=True,timeout=timeout);(OUT/(label+'.stdout')).write_text(z.stdout);(OUT/(label+'.stderr')).write_text(z.stderr)
    if z.returncode:raise RuntimeError(f'{label} rc={z.returncode}: {z.stderr[-4000:]}')
    return z.stdout
def dependencies(md):return md.read_text().replace('\\\n',' ').split(':',1)[1].split()
def main():
    mf=D/'SOURCE_MANIFEST.json';assert sha(mf)==os.environ['DEEP_PD_NATIVE_MANIFEST_SHA256']
    for directory,expected in [(D,None),(V2,'e153a7ccec63d0c741afde61bd3a94bcc3e5a2f1d4fbad957effc4923811e829'),(BASE,'fc7f1c16a80bf9050c8eb4981901fd34131bbddbe35038f99a1cbb28e1159619')]:
        f=directory/'SOURCE_MANIFEST.json'
        if expected:assert sha(f)==expected
        assert all(sha(directory/n)==h for n,h in json.loads(f.read_bytes())['files_SHA256'].items())
    assert sha(O2/'CPU-build.json')=='93ef932aeb6cd562ec7d01a2c0bc29f4776585d353d1ae3b4cde4bfd6a97c8a8'
    assert sha(O2/'libDeep_PD_native_r382.so')==R2['DSO_SHA256']
    OUT.mkdir(exist_ok=False);replacement={};old_objects=list(C['objects']);reused={}
    for n,h in C['objects'].items():assert sha(n)==h
    for contract in [R1,R2]:
        assert all(sha(n)==h for n,h in contract['project_dependencies_SHA256'].items())
        for entry in contract['objects']:
            p=pathlib.Path(entry['object']);assert sha(p)==entry['SHA256'];assert sha(p.with_suffix('.d'))==entry['MD_SHA256'];reused[str(p)]=entry['SHA256']
            if p.name.startswith('host-'):replacement[old_objects[int(p.stem.split('-')[1])]]=str(p)
    replacement[next(x for x in old_objects if x.endswith('/Deep-native-context-guard-r180d/2.o'))]=str(O2/'modern_native_engine_probe_pd_r382.cpp.o')
    base=C['guard_compiled'][2]['compile_argv'];base=base[:base.index('-MD')];prefix=C['private_parent_mirror']
    def conv(a):
        if a.startswith('-I/refactor/'):
            path=a[2:];relative=path[len(prefix):]if path.startswith(prefix+'/')else path.removeprefix('/refactor')
            return '-I'+str(O1/'project-mirror'/relative.lstrip('/'))
        return a
    flags=[conv(x)for x in base];obj=OUT/'retain_fmt_abi.o';argv=flags+['-MD','-MF',str(obj.with_suffix('.d')),'-c',str(D/'retain_fmt_abi.cpp'),'-o',str(obj)];run(argv,'retain-fmt-compile')
    expected='_ZNK3fmt3v1217basic_format_argsINS0_7contextEE3getEi'
    nm=run(['nm','--defined-only',str(obj)],'retain-fmt-symbols');assert any(line.split()[-2:]==['W',expected]for line in nm.splitlines())
    library=OUT/'libDeep_PD_native_r382.so';link=[replacement.get(x,x)for x in C['link_argv']];link[link.index('-o')+1]=str(library)
    archive=O1/'deep_pd_archive_r382.cpp.o';codec=O2/'deep_pd_codec_r382.cpp.o';link += [str(archive),str(codec),str(obj)];run(link,'link')
    exports=lambda p,label:{x.split()[-1]for x in run(['nm','-D','--defined-only',str(p)],label).splitlines()}
    old=exports(C['parent_library'],'old-symbols');new=exports(library,'new-symbols');assert old<=new,sorted(old-new)
    required=[x.split('(')[0].split()[-1]for x in(V2/'deep_pd_abi_r382.h').read_text().splitlines()if x.startswith(('int deep_','uint64_t deep_','const char *deep_'))];assert set(required)<=new
    ldd=run(['ldd',str(library)],'ldd');assert 'not found'not in ldd
    test=OUT/'CPU_archive_test';a=[x for x in flags if not x.startswith(('-L','-l','-Wl,'))];a[1:1]=['-I'+str(V2),'-I'+str(BASE)]
    a += [str(D/'CPU_archive_test.cpp'),str(archive),'-o',str(test)];run(a,'compile-CPU-archive');checks=json.loads(run([str(test)],'CPU-archive-test',240));assert checks['passed']
    objects={replacement.get(x,x):sha(replacement.get(x,x))for x in C['objects']};objects.update({str(p):sha(p)for p in[archive,codec,obj]});assert len(objects)==52
    result.update(completed=True,passed=True,source_manifest_SHA256=sha(mf),HOST_TUs_compiled=1,retainer_compile_argv=argv,
        retainer_dependencies_SHA256={x:sha(x)for x in dependencies(obj.with_suffix('.d'))},reused_CPU765_766_objects_SHA256=reused,
        library=str(library),library_SHA256=sha(library),all_objects_SHA256=objects,link_argv=link,all_original_dynamic_exports_retained=True,
        retained_fmt_weak_symbol=expected,ABI_symbols=required,ABI_header_SHA256=sha(V2/'deep_pd_abi_r382.h'),CPU_archive_test=checks,
        scope='Original dynamic exports fully retained via same-header weak template instantiation; only1 tiny HOST TU compiled. No Native dlopen/Runtime/GPU state/Model handoff/performance qualification.')
if OUT.exists():print(json.dumps({'completed':True,'passed':False,'error':'immutable output exists'}));raise SystemExit(1)
try:main()
except Exception as e:result.update(completed=True,error=str(e),traceback=traceback.format_exc())
finally:
    result['seconds']=time.monotonic()-START;OUT.mkdir(parents=True,exist_ok=True);(OUT/'CPU-build.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result.get(k)for k in['completed','passed','seconds','actual_step','error','library_SHA256']}))
    if not result['passed']:raise SystemExit(1)
