"""Two private implementation TUs + CPU recording test; never dlopen or create a GPU Context."""
from pathlib import Path
import datetime,hashlib,json,os,shlex,subprocess,time,traceback
D=Path(__file__).resolve().parent;B=Path('/refactor/candidates/Deep-fixed-Gemm-Cast-graph-r399-cpu-v3')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
START=time.monotonic()
q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],CPU_only=True,GPU_initialized=False,Native_dlopen=False,Model_loaded=False,private_implementation_TUs=2,existing_Native_or_shared_SDK_rebuilt=False,commands=[],compiled=[])
def run(args,label,seconds=160):
    left=255-(time.monotonic()-START)
    if left<=0:raise TimeoutError('r399 original CPU budget')
    z=subprocess.run([str(x)for x in args],capture_output=True,text=True,timeout=min(left,seconds));p=B/(label+'.log');p.write_text(z.stdout+z.stderr);q['commands'].append(dict(label=label,argv=[str(x)for x in args],rc=z.returncode,log=str(p),SHA256=sha(p)))
    if z.returncode:raise RuntimeError(label+' failed: '+z.stderr[-5000:])
    return z.stdout
def deps(path):return set(shlex.split(Path(path).read_text().replace('\\\n',' ').split(':',1)[1]))
if B.exists():raise FileExistsError("Immutable original r399 CPU output already exists")
try:
    B.mkdir(exist_ok=False)
    assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')
    mf=D/'SOURCE_MANIFEST.json';assert sha(mf)==os.environ['DEEP399_SOURCE_SHA256'];m=json.loads(mf.read_bytes());assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())
    c=json.loads((D/'BUILD_CONTRACT.json').read_bytes());assert all(sha(p)==h for p,h in c['dependencies_SHA256'].items())
    q['source_manifest_SHA256']=sha(mf)
    reuse=json.loads((D/'REUSE_CPU793.json').read_bytes());assert sha(reuse['CPU_receipt'])==reuse['CPU_receipt_SHA256'];prior=json.loads(Path(reuse['CPU_receipt']).read_bytes());unit=prior['compiled'][0];assert prior['actual_CPU']==reuse['CPU_actual'] and not prior['passed'] and unit['source'].endswith('/provider.cu') and unit['M_equals_MD'];assert sha(unit['object'])==unit['object_SHA256']==reuse['successful_provider_object_SHA256'] and sha(unit['MD'])==unit['MD_SHA256']==reuse['successful_provider_MD_SHA256'];assert sha(D/'provider.cu')==unit['source_SHA256'] and all(sha(p)==h for p,h in unit['dependencies_SHA256'].items());unit=dict(unit,reused_successful_subscope_from=prior['actual_CPU']);q['compiled'].append(unit);q.update(GPU_TUs_compiled_this_run=0,HOST_TUs_compiled_this_run=1,reused_provider_from_failed_CPU=prior['actual_CPU'])
    for name,flags in [('runtime_fixed_graph.cpp',c['HOST_prefix'])]:
        obj=B/(name+'.o');md=obj.with_suffix('.d');fresh=obj.with_suffix('.M');base=flags+['-I'+str(D)]
        run(base+['-M','-MF',str(fresh),'-MT',str(obj),str(D/name)],name+'-M',45);before={p:sha(p)for p in deps(fresh)}
        run(base+['-MD','-MF',str(md),'-c',str(D/name),'-o',str(obj)],name+'-compile',160)
        assert deps(md)==set(before)and all(sha(p)==h for p,h in before.items())
        q['compiled'].append(dict(source=str(D/name),source_SHA256=sha(D/name),object=str(obj),object_SHA256=sha(obj),MD=str(md),MD_SHA256=sha(md),dependencies_SHA256=before,M_equals_MD=True))
    library=B/'libDeep_fixed_Gemm_Cast_graph_r399.so';args=c['link_prefix']+[x['object']for x in q['compiled']]+c['link_libraries']+['-o',str(library)];run(args,'link',50)
    text=run(['nm','-D','--defined-only',str(library)],'nm',20);exports={x.split()[-1]for x in text.splitlines()if x.split()};required=c['required_CAPI'];assert set(required)<=exports
    dynamic=run(['readelf','--wide','-d',str(library)],'dynamic',15)
    ldd=run(['ldd','-r',str(library)],'ldd-r',30);assert 'not found'not in ldd and'undefined symbol'not in ldd
    resolved={}
    for line in ldd.splitlines():
        t=line.strip().split()
        if not t:continue
        path=t[2]if len(t)>2 and t[1]=='=>'else t[0]
        if path.startswith('/')and Path(path).is_file():resolved[path]=sha(path)
    assert str(c['SDK_library'])in resolved and resolved[c['SDK_library']]==c['dependencies_SHA256'][c['SDK_library']]
    test=B/'check_contract_CPU';run(['g++','-std=c++17','-O2','-pthread','-I'+str(D),str(D/'check_contract_CPU.cpp'),'-o',str(test)],'recording-compile',30)
    checks=json.loads(run([str(test)],'actual-shared-contract-recording',15));assert checks['passed']and not checks['GPU']
    q.update(completed=True,passed=True,library=str(library),library_SHA256=sha(library),required_CAPI=required,resolved_library_SHA256=resolved,DT_NEEDED=[x.split('[')[1].split(']')[0]for x in dynamic.splitlines()if'(NEEDED)'in x],recording_contract=checks,qualification='Private provider/Runtimebridge compile/link/MD/exports and actual shared lifecycle/recordtransaction with recording callbacks. No actual Runtime Context, vendor preparation, graph capture, numerical or GPU qualification.',GPU_GO=False)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
finally:
    q['seconds']=time.monotonic()-START;q['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    if B.exists():(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n')
    print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_CPU','error','seconds','library_SHA256']}),flush=True)
if not q['passed']:raise SystemExit(1)
