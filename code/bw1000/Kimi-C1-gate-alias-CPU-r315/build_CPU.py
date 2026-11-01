"""Two isolated HOST TUs and original Native closure; no dlopen/GPU/weights."""
from pathlib import Path
import json,hashlib,os,shlex,subprocess,time
D=Path(__file__).resolve().parent;R=Path('/refactor');B=R/'candidates/Kimi-C1-gate-alias-CPU-r315'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def semantic(args):
    result=[args[0]];i=1
    while i<len(args):
        v=args[i]
        if v in ['-o','-MF','-MT','-MQ','-MJ']:i+=2;continue
        if v in ['-c','-MD','-MMD','-MP','-MG','-M','-MM','-shared']:i+=1;continue
        if v.startswith(('-MF','-MT','-MQ','-MJ','-Wl,','-L','-l'))or v.endswith(('.cpp','.cc','.cxx','.o','.so','.a')):i+=1;continue
        result.append(v);i+=1
    return result


def main():
    assert all(os.environ.get(k)==''for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
    assert not os.environ.get('LD_PRELOAD') and not B.exists();B.mkdir();start=time.monotonic()
    m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert sha(D/'SOURCE_MANIFEST.json')==os.environ['KIMI_R315_MANIFEST_SHA']
    assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())
    q={'completed':False,'passed':False,'CPU_only':True,'GPU':0,'Model':0,'weights_loaded':0,'Native_dlopen':False,
       'source_manifest_SHA256':sha(D/'SOURCE_MANIFEST.json'),'actual_CPU':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'logs':{}}
    def run(args,label,cap):
        left=330-(time.monotonic()-start);assert left>0
        z=subprocess.run(args,capture_output=True,text=True,timeout=min(cap,left));p=B/(label+'.log');p.write_text(z.stdout+z.stderr)
        q['logs'][label]={'rc':z.returncode,'SHA256':sha(p)}
        if z.returncode:raise RuntimeError(label+' failed; retained log '+str(p))
        return z.stdout
    try:
        from patch_gate_alias import patch
        oldN=R/'incoming/Kimi-append512-r247/native';N=B/'native';N.mkdir()
        index=json.loads((D/'baseline-native-index.json').read_bytes())
        for name,h in index.items():
            assert name.startswith('native/');name=name[len('native/'):];p=oldN/name;assert sha(p)==h
            dest=N/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        (N/'delta.cpp').write_bytes(patch((oldN/'delta.cpp').read_text()).encode())
        (N/'gate_alias_r314.hpp').write_bytes((D/'gate_alias_r314.hpp').read_bytes())
        ctxp=R/'incoming/Kimi-ctx2048-r123e/per-TU-original-context-r123e.json'
        assert sha(ctxp)=='07bb301b870f90e1ce9e8f771b2c73080937ee0d200ef753a7521fa81f922454'
        ctx=json.loads(ctxp.read_bytes());flags=semantic(ctx['delta']['argv'])
        prefix=['-I'+str(N),'-I'+str(N/'engine'),'-I/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/engine']
        obj=B/'delta.o';run([flags[0],*prefix,*flags[1:],'-MD','-MF',str(B/'delta.d'),'-c',str(N/'delta.cpp'),'-o',str(obj)],'delta-HOST',120)
        run([flags[0],*prefix,*flags[1:],'-M','-MF',str(B/'delta.fresh.d'),'-MT',str(obj),str(N/'delta.cpp')],'delta-fresh-MD',30)
        deps=lambda p:set(shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1]))
        md=deps(B/'delta.d');assert md==deps(B/'delta.fresh.d')
        base179p=R/'candidates/Kimi-long-agent32K-r179/CPU-build.json'
        assert sha(base179p)=='37ac44052ffb3b0e43cf7e440178aad4096f09dcd97a8b12d6eca6fc65cd827d'
        oldmd=json.loads(base179p.read_bytes())['all_new_MD']['delta'];old179='/refactor/incoming/Kimi-long-agent32K-r179/native'
        normalized={p.replace(str(N),old179,1):p for p in md};extra={old179+'/gate_alias_r314.hpp'}
        assert set(normalized)-set(oldmd)==extra and not(set(oldmd)-set(normalized))
        assert all(sha(actual)==oldmd[name]for name,actual in normalized.items()if name not in extra|{old179+'/delta.cpp'})
        basep=R/'candidates/Kimi-append512-r247/CPU-build.json';assert sha(basep)=='21fc8ed69c6db1ec3a3c8925e61dc5f73112b5fea596a8caef977008da57046b'
        base=json.loads(basep.read_bytes());assert base['completed']
        for p,h in base['all_linked_object_SHA256'].items():assert sha(p)==h
        lbp=R/'candidates/Kimi-W2-pair-r9280/CPU-build.json';assert sha(lbp)=='e86d3428b6134873f5525cbfd0409168593ec1d17c2fa3b48bfcebfa46ebf86e'
        lb=json.loads(lbp.read_bytes());link=next(c[:]for c in lb['commands']if '-shared'in c and '-o'in c and c[c.index('-o')+1]==lb['native'])
        mapping={
          '/refactor/candidates/Kimi-speculative-batch-Model-r9122c/pair_probe.o':'/refactor/candidates/Kimi-append512-r247/pair_probe.o',
          '/refactor/candidates/Kimi-speculative-batch-Model-r9122c/worker.o':'/refactor/candidates/Kimi-long-agent32K-r179/worker.o',
          '/refactor/candidates/Kimi-speculative-batch-Model-r9122c/delta.o':'/refactor/candidates/Kimi-long-agent32K-r179/delta.o',
          '/refactor/candidates/Kimi-speculative-batch-Model-r9122c/trace.o':'/refactor/candidates/Kimi-long-agent32K-r179/trace.o',
          '/refactor/candidates/kimi-compact-cache-r4974/objects/infer_engine.o':'/refactor/candidates/Kimi-long-agent32K-r179/infer_engine.o',
          '/refactor/candidates/kimi-compact-cache-r4974/objects/kimi_k3_allocate_cache.o':'/refactor/candidates/Kimi-long-agent32K-r179/allocate.o',
          '/refactor/candidates/Kimi-I192-batch-Model-r9129/moe.o':'/refactor/candidates/Kimi-long-scale-exact-r201/moe.o'}
        for a,b in mapping.items():assert link.count(a)==1;link[link.index(a)]=b
        assert {p for p in link if p.endswith('.o')}==set(base['all_linked_object_SHA256'])
        olddelta='/refactor/candidates/Kimi-long-agent32K-r179/delta.o';link[link.index(olddelta)]=str(obj)
        main=B/'libKimi_C1_gate_alias_Main_r315.so';link[link.index('-o')+1]=str(main)
        for i,v in enumerate(link):
            if v.startswith('-Wl,-Map'):link[i]='-Wl,-Map,'+str(B/'link.map')
        run(link,'Main-link',50)
        oldshimp=R/'results/Kimi-layer0-link-CPU-r9038e-910407.272.json';assert sha(oldshimp)==m['original_shim_receipt_SHA256']
        oldshim=json.loads(oldshimp.read_bytes());shim=oldshim['command'][:];shim[1:1]=['-I'+str(N),'-I'+str(D)]
        native=B/'libKimi_C1_gate_alias_shim_r315.so';shim[shim.index('-shared')+1]=str(D/'layer0_native.cpp')
        shim[shim.index(oldshim['actual678_DSO'])]=str(main);shim[shim.index('-o')+1]=str(native)
        run(shim,'shim-HOST-link',90);ldd=run(['ldd',str(native)],'ldd',20);assert 'not found'not in ldd
        q.update(completed=True,passed=True,new_HOST_TUs=2,new_GPU_TUs=0,Native_linked=True,
                 native=str(native),native_SHA256=sha(native),Main=str(main),Main_SHA256=sha(main),
                 candidate_Delta_SHA256=sha(N/'delta.cpp'),original_Delta_SHA256=sha(oldN/'delta.cpp'),
                 other_original_dependency_bytes_preserved=True,new_gate_alias_header_only=True,
                 unchanged_linked_objects={p:h for p,h in base['all_linked_object_SHA256'].items()if p!=olddelta},
                 actual_dependency_library_SHA256=base['actual_dependency_library_SHA256'],
                 fixture_manifest_SHA256=sha(D/'fixture_manifest.json'),qualification='HOST compile/link and dependency closure only; no Model/GPU/C1 numerics/performance qualification')
    except BaseException as e:q['error']=repr(e)
    finally:
        q['elapsed_s']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n')
        print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_CPU','error','elapsed_s','native_SHA256']}))
    assert q['completed'] and q['passed']

if __name__=='__main__':main()
