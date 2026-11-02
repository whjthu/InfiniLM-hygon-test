import concurrent.futures,hashlib,json,os,pathlib,subprocess,time,traceback
D=pathlib.Path(__file__).resolve().parent
BASE=D.parent
V1=pathlib.Path('/refactor/candidates/Deep-PD-native-r382-cpu-v1')
OUT=pathlib.Path('/refactor/candidates/Deep-PD-native-r382-cpu-v2')
C=json.loads((BASE/'BUILD_CONTRACT.json').read_bytes())
REUSE=json.loads((D/'REUSE_CPU765.json').read_bytes())
START=time.monotonic()
RESULT={'completed':False,'passed':False,'CPU_only':True,'GPU_initialized':False,'Native_dlopen':False,'Model_loaded':False,'GPU_TUs_compiled':0,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def run(argv,label,timeout=240):
    z=subprocess.run([str(x)for x in argv],capture_output=True,text=True,timeout=timeout)
    (OUT/(label+'.stdout')).write_text(z.stdout);(OUT/(label+'.stderr')).write_text(z.stderr)
    if z.returncode:raise RuntimeError(f'{label} rc={z.returncode}: {z.stderr[-5000:]}')
    return z.stdout
def deps(md):return md.read_text().replace('\\\n',' ').split(':',1)[1].split()
def flags(q):
    argv=q['compile_argv'];argv=argv[:argv.index('-MD')]
    prefix=C['private_parent_mirror']
    def conv(a):
        if a.startswith('-I/refactor/'):
            path=a[2:];rel=path[len(prefix):]if path.startswith(prefix+'/')else path.removeprefix('/refactor')
            return '-I'+str(V1/'project-mirror'/rel.lstrip('/'))
        return a
    argv=[conv(x)for x in argv];argv[1:1]=['-I'+str(D),'-I'+str(BASE),'-I'+str(BASE/'headers')]
    return argv
def main():
    mf=D/'SOURCE_MANIFEST.json';assert sha(mf)==os.environ['DEEP_PD_NATIVE_MANIFEST_SHA256']
    manifest=json.loads(mf.read_bytes());assert all(sha(D/n)==h for n,h in manifest['files_SHA256'].items())
    assert sha(BASE/'SOURCE_MANIFEST.json')==manifest['parent_v1_manifest_SHA256']
    assert all(sha(BASE/n)==h for n,h in json.loads((BASE/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items())
    assert sha(V1/'CPU-build.json')==REUSE['receipt_SHA256']
    previous=json.loads((V1/'CPU-build.json').read_bytes());assert previous['completed']and not previous['passed']and 'const void*'in previous['error']
    OUT.mkdir(exist_ok=False)
    replacement={};reused=[]
    objects=list(C['objects'])
    for p,h in C['objects'].items():assert sha(p)==h
    assert all(sha(n)==h for n,h in REUSE['project_dependencies_SHA256'].items())
    for entry in REUSE['objects']:
        p=pathlib.Path(entry['object']);assert sha(p)==entry['SHA256'];assert sha(p.with_suffix('.d'))==entry['MD_SHA256']
        if p.name.startswith('host-'):
            index=int(p.stem.split('-')[1]);replacement[objects[index]]=str(p)
        all_deps={n:sha(n)for n in deps(p.with_suffix('.d'))}
        reused.append({**entry,'all_dependencies_SHA256':all_deps})
    jobs=[]
    capi=next(x for x in C['guard_compiled']if x['object'].endswith('/2.o'))
    for name,old in [('modern_native_engine_probe_pd_r382.cpp',capi['object']),('deep_pd_codec_r382.cpp',None)]:
        obj=OUT/(name+'.o');argv=flags(capi)
        argv+=['-MD','-MF',str(obj.with_suffix('.d')),'-c',str(D/name),'-o',str(obj)]
        jobs.append((old,obj,argv,D/name))
    def compile_one(item):
        old,obj,argv,source=item;run(argv,obj.stem)
        all_deps={n:sha(n)for n in deps(obj.with_suffix('.d'))}
        return {'replaces':old,'object':str(obj),'object_SHA256':sha(obj),'source':str(source),'source_SHA256':sha(source),'compile_argv':argv,'dependencies_SHA256':all_deps}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:compiled=list(pool.map(compile_one,jobs))
    replacement.update({x['replaces']:x['object']for x in compiled if x['replaces']})
    link=[replacement.get(x,x)for x in C['link_argv']];library=OUT/'libDeep_PD_native_r382.so';link[link.index('-o')+1]=str(library)
    archive=V1/'deep_pd_archive_r382.cpp.o';link+=[str(archive),compiled[1]['object']]
    run(link,'link')
    exports=lambda p,label:{line.split()[-1]for line in run(['nm','-D','--defined-only',str(p)],label).splitlines()}
    old=exports(C['parent_library'],'old-symbols');new=exports(library,'new-symbols');assert old<=new,sorted(old-new)
    required=[line.split('(')[0].split()[-1]for line in(D/'deep_pd_abi_r382.h').read_text().splitlines()if line.startswith(('int deep_','uint64_t deep_','const char *deep_'))]
    assert set(required)<=new,sorted(set(required)-new)
    ldd=run(['ldd',str(library)],'ldd');assert 'not found'not in ldd
    test=OUT/'CPU_archive_test';argv=[x for x in flags(capi)if not x.startswith(('-L','-l','-Wl,'))]
    argv+=[str(D/'CPU_archive_test.cpp'),str(archive),'-o',str(test)];run(argv,'compile-CPU-archive')
    tests=json.loads(run([str(test)],'CPU-archive-test',300));assert tests['passed']
    all_objects={replacement.get(x,x):sha(replacement.get(x,x))for x in C['objects']}
    all_objects.update({str(archive):sha(archive),compiled[1]['object']:compiled[1]['object_SHA256']})
    assert len(all_objects)==51
    RESULT.update(completed=True,passed=True,source_manifest_SHA256=sha(mf),parent_v1_failed_CPU='910407.765',parent_v1_receipt_SHA256=REUSE['receipt_SHA256'],
        HOST_TUs_compiled=2,GPU_TUs_compiled=0,compiled=compiled,reused_CPU765_objects=reused,library=str(library),library_SHA256=sha(library),
        all_objects_SHA256=all_objects,link_argv=link,all_original_dynamic_exports_retained=True,ABI_symbols=required,CPU_archive_test=tests,
        scope='CPU765 successful immutable HOST objects reused with exact MD/project dependency SHA; only const codec fix and generation-poison CAPI rebuilt; no Native dlopen/GPU state/Model')
if OUT.exists():
    print(json.dumps({'completed':True,'passed':False,'error':'Immutable output exists; no files modified'}));raise SystemExit(1)
try:main()
except Exception as exc:RESULT.update(completed=True,error=str(exc),traceback=traceback.format_exc())
finally:
    RESULT['seconds']=time.monotonic()-START;OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'CPU-build.json').write_text(json.dumps(RESULT,indent=2)+'\n')
    print(json.dumps({k:RESULT.get(k)for k in ['completed','passed','seconds','actual_step','error']}))
    if not RESULT['passed']:raise SystemExit(1)
