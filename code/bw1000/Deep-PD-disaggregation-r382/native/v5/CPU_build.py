"""Qualify the existing CPU768 DSO; never compile/link Native or load it."""
import hashlib,json,os,pathlib,re,subprocess,time,traceback
D=pathlib.Path(__file__).resolve().parent;BASE=D.parent;V2=BASE/'v2';V4=BASE/'v4'
OUT=pathlib.Path('/refactor/candidates/Deep-PD-native-r382-cpu-v5')
ART=pathlib.Path('/refactor/candidates/Deep-PD-native-r382-cpu-v4')
O1=pathlib.Path('/refactor/candidates/Deep-PD-native-r382-cpu-v1')
O2=pathlib.Path('/refactor/candidates/Deep-PD-native-r382-cpu-v2')
START=time.monotonic();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(pathlib.Path(p).read_bytes())
step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
result={'completed':False,'passed':False,'CPU_only':True,'GPU_initialized':False,'Native_dlopen':False,'Model_loaded':False,'GPU_TUs_compiled':0,'HOST_Native_TUs_compiled':0,'Native_link_executed':False,'actual_step':step}
def run(a,label,timeout=240):
    z=subprocess.run([str(x)for x in a],capture_output=True,text=True,timeout=timeout);(OUT/(label+'.stdout')).write_text(z.stdout);(OUT/(label+'.stderr')).write_text(z.stderr)
    if z.returncode:raise RuntimeError(f'{label} rc={z.returncode}: {z.stderr[-4000:]}')
    return z.stdout
def depnames(p):return p.read_text().replace('\\\n',' ').split(':',1)[1].split()
def main():
    mf=D/'SOURCE_MANIFEST.json';assert sha(mf)==os.environ['DEEP_PD_NATIVE_MANIFEST_SHA256']
    refs=[]
    for version in ['v1','v2','v3','v4','v5']:
        f=BASE/('SOURCE_MANIFEST.json'if version=='v1'else version+'/SOURCE_MANIFEST.json');v=load(f)
        assert all(sha(f.parent/n)==h for n,h in v['files_SHA256'].items())
        refs.append({'path':str(f),'SHA256':sha(f)})
    inputs=load(D/'INPUT_ARTIFACTS.json');library=pathlib.Path(inputs['DSO']);assert sha(library)==inputs['DSO_SHA256']
    assert sha(ART/'CPU-build.json')==inputs['receipt_SHA256']
    OUT.mkdir(exist_ok=False);C=load(BASE/'BUILD_CONTRACT.json');R1=load(V2/'REUSE_CPU765.json');R2=load(V4/'REUSE_CPU766.json')
    exports=lambda p,label:{x.split()[-1]for x in run(['nm','-D','--defined-only',str(p)],label).splitlines()}
    old=exports(C['parent_library'],'old-symbols');new=exports(library,'new-symbols');assert old<=new
    header=V2/'deep_pd_abi_r382.h';required=re.findall(r'\b(deep_pd_state_\w+)\s*\(',header.read_text())
    assert set(required)==set(inputs['required'])and len(required)==11 and set(required)<=new
    assert not inputs['missing']
    for n,h in C['objects'].items():assert sha(n)==h
    old_objects=list(C['objects']);replacement={};added={};receipts=[]
    for contract in [R1,R2]:
        assert all(sha(n)==h for n,h in contract['project_dependencies_SHA256'].items())
        for e in contract['objects']:
            p=pathlib.Path(e['object']);assert sha(p)==e['SHA256']and sha(p.with_suffix('.d'))==e['MD_SHA256']
            if p.name.startswith('host-'):replacement[old_objects[int(p.stem.split('-')[1])]]=(str(p),'910407.765')
            elif p.name=='modern_native_engine_probe_pd_r382.cpp.o':
                original=next(x for x in old_objects if x.endswith('/Deep-native-context-guard-r180d/2.o'));replacement[original]=(str(p),'910407.766')
            else:added[str(p)]='910407.765'if p.parent==O1 else'910407.766'
    retainer=pathlib.Path(inputs['retainer']['object']);assert sha(retainer)==inputs['retainer']['SHA256']and sha(retainer.with_suffix('.d'))==inputs['retainer']['MD_SHA256']
    assert all(sha(n)==h for n,h in inputs['retainer']['project_dependencies_SHA256'].items());added[str(retainer)]='910407.768'
    objects={replacement.get(x,(x,None))[0]:sha(replacement.get(x,(x,None))[0])for x in old_objects};objects.update({p:sha(p)for p in added});assert len(objects)==52
    def provenance(path,compiled_step,original=None):
        p=pathlib.Path(path);md=p.with_suffix('.d');deps=depnames(md);source=deps[0]
        e={'object':path,'object_SHA256':sha(p),'source':source,'source_SHA256':sha(source),'MD':str(md),'MD_SHA256':sha(md),
           'dependencies_SHA256':{n:sha(n)for n in deps},'compile_step':compiled_step,
           'dependency_observation':'exact compiler MD paths, readonly SHA recheck in same fixed Linux container during qualification'}
        if original:e['old_object']=original
        return e
    replacements=[provenance(p,s,old)for old,(p,s)in replacement.items()];new_objects=[provenance(p,s)for p,s in added.items()]
    for version,known in [('v1','910407.765'),('v2','910407.766'),('v3','910407.767'),('v4','910407.768')]:
        p=pathlib.Path('/refactor/candidates/Deep-PD-native-r382-cpu-'+version+'/CPU-build.json');v=load(p);receipts.append({'path':str(p),'SHA256':sha(p),'actual_CPU':v.get('actual_step')or known,'passed':v['passed']})
    link=[replacement.get(x,(x,None))[0]for x in C['link_argv']];link[link.index('-o')+1]=str(library);link += [str(O1/'deep_pd_archive_r382.cpp.o'),str(O2/'deep_pd_codec_r382.cpp.o'),str(retainer)]
    base=C['guard_compiled'][2]['compile_argv'];base=base[:base.index('-MD')];prefix=C['private_parent_mirror']
    def conv(a):
        if a.startswith('-I/refactor/'):
            path=a[2:];relative=path[len(prefix):]if path.startswith(prefix+'/')else path.removeprefix('/refactor')
            return '-I'+str(O1/'project-mirror'/relative.lstrip('/'))
        return a
    # Compile only the host CPU archive consumer; no Native/retainer/link redo.
    test=OUT/'CPU_archive_test';argv=[conv(x)for x in base if not x.startswith(('-L','-l','-Wl,'))];argv[1:1]=['-I'+str(V2),'-I'+str(BASE)]
    argv += [str(V4/'CPU_archive_test.cpp'),str(O1/'deep_pd_archive_r382.cpp.o'),'-o',str(test)];run(argv,'compile-CPU-consumer')
    checks=json.loads(run([str(test)],'CPU-archive-test',240));assert checks['passed']
    ldd=run(['ldd',str(library)],'ldd');assert 'not found'not in ldd
    graph={'completed':True,'passed':True,'CPU_only':True,'Native_dlopen':False,'GPU':0,'Model':False,'parent_objects_SHA256':C['objects'],'all_objects_SHA256':objects,
           'replacements':replacements,'added_objects':new_objects,'build_receipts':receipts,'native_source_manifests':refs,'library':str(library),'library_SHA256':sha(library),
           'ABI_header_SHA256':sha(header),'link_argv':link,'link_provenance':'exact frozen CPU768 source recipe; not re-executed by qualifier','qualification_CPU':step}
    graph_path=OUT/'Native-object-graph.json';graph_path.write_text(json.dumps(graph,indent=2)+'\n')
    result.update(completed=True,passed=True,source_manifest_SHA256=sha(mf),library=str(library),library_SHA256=sha(library),all_objects_SHA256=objects,
        all_original_dynamic_exports_retained=True,ABI_symbols=required,ABI_header_SHA256=sha(header),CPU_archive_test=checks,
        object_graph=str(graph_path),object_graph_SHA256=sha(graph_path),link_argv=link,retainer_compile_step='910407.768',
        scope='Readonly qualification of already-linked CPU768 DSO; no Native/retainer compilation or link. Actual CPU archive consumer only; no Runtime/GPU state/Model handoff/performance.')
if OUT.exists():print(json.dumps({'completed':True,'passed':False,'error':'immutable output exists'}));raise SystemExit(1)
try:main()
except Exception as e:result.update(completed=True,error=str(e),traceback=traceback.format_exc())
finally:
    result['seconds']=time.monotonic()-START;OUT.mkdir(parents=True,exist_ok=True);(OUT/'CPU-build.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result.get(k)for k in['completed','passed','seconds','actual_step','error','library_SHA256','object_graph_SHA256']}))
    if not result['passed']:raise SystemExit(1)
