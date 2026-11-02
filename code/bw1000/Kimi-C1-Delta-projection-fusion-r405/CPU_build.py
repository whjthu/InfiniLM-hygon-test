"""One private HOST TU + existing current Native dependency. No Native dlopen/GPU/weights."""
from pathlib import Path
import os,json,hashlib,subprocess,shlex,time,re,sys
D=Path(__file__).resolve().parent;R=Path('/refactor');B=R/'candidates/Kimi-C1-Delta-projection-fusion-r405'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    start=time.monotonic();assert not B.exists();B.mkdir()
    q=dict(completed=False,passed=False,CPU_only=True,Native_dlopen=False,GPU=0,Model=0,new_HOST_TUs=1,new_GPU_TUs=0,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],commands=[],logs={})
    def run(argv,label,cap):
        left=260-(time.monotonic()-start);assert left>0
        z=subprocess.run(argv,text=True,capture_output=True,timeout=min(left,cap));p=B/(label+'.log');p.write_text(z.stdout+z.stderr);q['commands'].append(argv);q['logs'][label]=dict(rc=z.returncode,path=str(p),SHA256=sha(p))
        if z.returncode:raise RuntimeError(label+' failed: '+str(p))
        return z.stdout
    try:
        assert all(os.environ.get(k) in ['', '-1'] for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
        mf=D/'SOURCE_MANIFEST.json';assert sha(mf)==os.environ['KIMI405_SOURCE_SHA256'];m=json.loads(mf.read_bytes());assert all(sha(D/n)==h for n,h in m['files_SHA256'].items());q['source_manifest_SHA256']=sha(mf)
        pins=json.loads((D/'SOURCE_CARD.json').read_bytes())
        for p,h in pins['actual_dependencies_SHA256'].items():assert sha(p)==h,p
        oldN=R/'incoming/Kimi-append512-r247/native';N=B/'native';N.mkdir()
        index=json.loads((D/'baseline-native-index.json').read_bytes())
        for name,h in index.items():
            assert name.startswith('native/');p=oldN/name[7:];assert sha(p)==h
            dest=N/name[7:];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        assert sha(oldN/'delta.cpp')==pins['original_Delta_SHA256']
        for name in ['delta_r405.cpp','thin_consumer.inc','projection_r405.hpp']:(N/name).write_bytes((D/name).read_bytes())
        src=N/'delta_r405.cpp';obj=B/'delta_projection.o'
        ctx=json.loads((R/'incoming/Kimi-ctx2048-r123e/per-TU-original-context-r123e.json').read_bytes())['delta']['argv']
        flags=[];i=1
        while i<len(ctx):
            a=ctx[i]
            if a in ['-MF','-o']:i+=2;continue
            if a in ['-c','-MD'] or a.endswith('.cpp'):i+=1;continue
            flags.append(a);i+=1
        prefix=['-I'+str(N),'-I'+str(N/'engine'),'-I/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/engine']
        run([ctx[0],*prefix,*flags,'-MD','-MF',str(B/'delta_projection.d'),'-c',str(src),'-o',str(obj)],'one-HOST-compile',150)
        deps=shlex.split((B/'delta_projection.d').read_text().replace('\\\n',' ').split(':',1)[1]);assert all(Path(p).is_file() for p in deps)
        q['MD_source_and_dependencies_SHA256']={p:sha(p)for p in sorted(set(deps))};q['MD_SHA256']=sha(B/'delta_projection.d')
        oldmd=json.loads((R/'candidates/Kimi-long-agent32K-r179/CPU-build.json').read_bytes())['all_new_MD']['delta']
        for p,h in oldmd.items():
            if p.startswith('/refactor/incoming/Kimi-long-agent32K-r179/native/'):continue
            assert sha(p)==h,p
        q['original_actual_header_MD_preserved']=True
        prior=json.loads((R/'results/Kimi-layer0-link-CPU-r9038e-910407.272.json').read_bytes());link=prior['command'][:]
        link[link.index(prior['source_path'])]=str(obj);link[link.index(prior['actual678_DSO'])]=pins['current_Native_path']
        lib=B/'libKimi_Delta_projection_r405.so';link[link.index('-o')+1]=str(lib);link.insert(link.index('-o'),'-Wl,-Bsymbolic')
        run(link,'one-HOST-link',50)
        ldd=run(['ldd',str(lib)],'ldd',15);assert 'not found' not in ldd
        resolved={p for p in re.findall(r'(/[^\s()]+)',ldd) if Path(p).is_file()}
        assert str(Path(pins['current_Native_path']).resolve()) in {str(Path(p).resolve())for p in resolved}
        q['actual_resolved_dependency_library_SHA256']={p:sha(p)for p in sorted(resolved)}
        q['DT_NEEDED']=run(['readelf','-d',str(lib)],'ELF-DT-NEEDED',10)
        symbols=run(['nm','-D','--defined-only',str(lib)],'defined-exports',10)
        for name in ['layer0_create_r9038','layer0_batch_run_r9122','layer0_projection_mode_r405','layer0_projection_stats_r405']:assert any(line.split()[-1]==name for line in symbols.splitlines()),name
        demangled=run(['nm','-D','--defined-only','-C',str(lib)],'defined-CXX-exports',10)
        for name in ['KimiK3DeltaAttention::KimiK3DeltaAttention(', 'KimiK3DeltaAttention::forward(', 'KimiK3DeltaAttention::process_weights_after_loading(']:assert name in demangled,name
        checks=json.loads(run([sys.executable,'-B',str(D/'check_CPU.py')],'actual-consumer-CPU-recording',15));assert checks['completed'] and checks['passed'];q['CPU_consumer']=checks
        q.update(completed=True,passed=True,native=str(lib),native_SHA256=sha(lib),object=str(obj),object_SHA256=sha(obj),current_Native_path=pins['current_Native_path'],current_Native_SHA256=pins['actual_dependencies_SHA256'][pins['current_Native_path']],qualification='one private HOST compiled/linked and actual Python call schedule recording only; no device numerics/collective execution/Model proof')
    except BaseException as e:q['error']=repr(e)
    finally:
        q['elapsed_s']=time.monotonic()-start;p=B/'CPU-build.json';p.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_CPU','elapsed_s','native_SHA256','error']}))
    assert q['completed'] and q['passed']
if __name__=='__main__':main()
