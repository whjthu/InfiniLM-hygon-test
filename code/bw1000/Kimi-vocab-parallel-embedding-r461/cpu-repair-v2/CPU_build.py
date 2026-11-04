"""Two changed HOST TUs + one new GPU TU, 31 exact objects reused. No DSO load."""
from pathlib import Path
import ast,hashlib,json,os,posixpath,shlex,subprocess,sys,time,traceback
D=Path(__file__).resolve().parent;P=D.parent;N=D/'native';B=Path('/refactor/candidates/Kimi-vocab-parallel-embedding-r461-CPU-v2')
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def semantic(a):
    out=[a[0]];i=1
    while i<len(a):
        z=a[i]
        if z in ['-o','-MF','-MT','-MQ','-MJ']:i+=2;continue
        if z in ['-c','-MD','-MMD','-MP','-MG','-M','-MM','-shared']:i+=1;continue
        if z.startswith(('-MF','-MT','-MQ','-MJ','-Wl,','-L','-l'))or z.endswith(('.cpp','.cc','.cxx','.o','.so','.a')):i+=1;continue
        out.append(z);i+=1
    return out
def deps(p):return set(shlex.split(Path(p).read_text().replace('\\\n',' ').split(':',1)[1]))

def main():
    assert all(os.environ.get(k)==''for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
    assert not os.environ.get('LD_PRELOAD')and 'torch'not in sys.modules
    assert not B.exists();B.mkdir();start=time.monotonic()
    q=dict(completed=False,passed=False,CPU_only=True,GPU=0,Model=0,Native_dlopen=False,SDK_execution=False,
        checkpoint_weight_payload=False,new_HOST_TUs=2,new_GPU_TUs=1,pure_source_contract_TUs=0,
        actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],source_manifest_SHA256=H(D/'CPU_SOURCE_MANIFEST.json'),commands=[],logs={})
    def save():(B/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
    def run(args,label,cap=90):
        left=270-(time.monotonic()-start);assert left>0
        q['phase']=label;q['commands'].append(args);save()
        try:z=subprocess.run(args,capture_output=True,text=True,timeout=min(left,cap))
        except subprocess.TimeoutExpired as error:
            path=B/(label+'.log');path.write_bytes((error.stdout or b'')+(error.stderr or b''));q['logs'][label]=dict(timeout=True,SHA256=H(path));raise
        path=B/(label+'.log');path.write_text(z.stdout+z.stderr);q['logs'][label]=dict(rc=z.returncode,SHA256=H(path))
        if z.returncode:raise RuntimeError(label+' failed; original '+str(path))
        return z.stdout
    def receipt(pin):
        assert H(pin['path'])==pin['SHA256'];v=json.loads(Path(pin['path']).read_bytes())
        assert v['completed']and v['passed']and v['CPU_only']and not v['GPU']and not v['Native_dlopen'];return v
    try:
        m=json.loads((D/'CPU_SOURCE_MANIFEST.json').read_bytes())
        for p,h in m['files_SHA256'].items():assert H(D/p)==h,p
        for p in D.glob('*.py'):ast.parse(p.read_bytes())
        pins=json.loads((P/'PARENT_PINS.json').read_bytes());base=receipt(pins['CPU966_receipt']);parent=receipt(pins['CPU961_receipt'])
        assert H(base['native'])==base['native_SHA256']==pins['Native_parent']['SHA256']
        for p,h in base['all_linked_object_SHA256'].items():assert H(p)==h,p
        for pin in base['actual_dependency_library_SHA256'].values():assert H(pin['path'])==pin['SHA256']
        for p,h in json.loads((P/'SOURCE_INPUTS.json').read_bytes()).items():assert H(p)==h,p
        for p,h in pins['old_class_headers_actual_CPU961_head_MD'].items():
            actual={posixpath.normpath(k):v for k,v in parent['HOST']['head']['all_new_MD'].items()}
            assert actual[p]==H(p)==h,p
        q['old_class_header_ABI_source_pins']=pins['old_class_headers_actual_CPU961_head_MD']
        from check_source import check
        q['source_contract']=check();save()
        loader_parent=json.loads((P/'LOADER_PARENT.json').read_bytes())
        for key in ['loader','head_source']:assert H(loader_parent[key]['path'])==loader_parent[key]['SHA256']
        assert (P/'parameter_loader_r461.py').read_text().replace('from head_source_r461 import expected_from_CPU','from .shard_head_contract_r9183 import expected_from_CPU')==Path(loader_parent['loader']['path']).read_text()
        repair=json.loads((D/'PARENT.json').read_bytes());oldpin=repair['CPU979_receipt']
        assert H(oldpin['path'])==oldpin['SHA256']
        previous=json.loads(Path(oldpin['path']).read_bytes())
        assert previous['actual_CPU']=='910407.979'and not previous['passed']and previous['phase']=='pair_probe-compile'
        assert not previous['Native_dlopen']and not previous['SDK_execution']and previous['GPU']==0
        for label in ['actual-Weight-Embedding-source-recording-contract','actual-loader-main-recording-contract']:
            oldlog=Path(oldpin['path']).parent/(label+'.log')
            assert H(oldlog)==previous['logs'][label]['SHA256']and previous['logs'][label]['rc']==0
        assert previous['actual_source_CPU_contract']['passed']and previous['actual_Python_loader_contract']['passed']
        q['reused_CPU979_contracts']=dict(receipt=oldpin,actual_source_CPU_contract=previous['actual_source_CPU_contract'],actual_Python_loader_contract=previous['actual_Python_loader_contract'],recording_source_identity=previous['recording_source_identity'],executed_again=False)
        save()
        # Exact successful CPU869 GPU compile flags, same gfx936/Runtime/SDK ABI;
        # no old GEMV source, object, kernel or math is linked into this Native.
        gp=pins['GPU_toolchain'];assert H(gp['receipt'])==gp['receipt_SHA256']
        gr=json.loads(Path(gp['receipt']).read_bytes());assert gr['passed']and gr['actual_CPU']=='910407.869'
        for p,h in gp['libraries_SHA256'].items():assert H(p)==h,p
        gpu_argv=next(c['argv']for c in gr['commands']if c['label']=='gemv.cu-compile')
        gpu_prefix=gpu_argv[:gpu_argv.index('-MD')]
        old_include='-I/refactor/incoming/Kimi-lowrank-M1-GEMV-r432';assert old_include in gpu_prefix
        gpu_prefix=[v for v in gpu_prefix if v!=old_include]+['-I'+str(P)]
        assert '-arch=gfx936'in gpu_prefix and not any('fast-math'in x for x in gpu_prefix)
        compile_units=[]
        pair_cmd=next(a for a in base['commands']if '-c'in a and Path(a[a.index('-c')+1]).name=='pair_probe.cpp')
        head_cmd=next(a for a in parent['commands']if '-c'in a and Path(a[a.index('-c')+1]).name=='kimi_shard_model_r9183.cpp')
        compile_units.append(('pair_probe',N/'pair_probe.cpp',['g++','-I'+str(N),'-I'+str(P),*semantic(pair_cmd)[1:]]))
        compile_units.append(('text_model',P/'native/kimi_k3_text_model.cpp',['g++','-I'+str(N),'-I'+str(P),'-I/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/models/kimi_k3',*semantic(head_cmd)[1:]]))
        compile_units.append(('masked_embedding',P/'masked_embedding.cu',gpu_prefix))
        objects={};details={}
        for unit,source,prefix in compile_units:
            before=B/(unit+'.before.d');actual=B/(unit+'.d');obj=B/(unit+'.o')
            run([*prefix,'-M','-MF',str(before),'-MT',str(obj),str(source)],unit+'-before-M',30)
            snapshot={p:H(p)for p in sorted(deps(before))}
            run([*prefix,'-MD','-MF',str(actual),'-c',str(source),'-o',str(obj)],unit+'-compile',90)
            assert deps(actual)==set(snapshot)and all(H(p)==h for p,h in snapshot.items())
            objects[unit]=str(obj);details[unit]=dict(source=str(source),source_SHA256=H(source),object_SHA256=H(obj),all_new_MD=snapshot,MD_before_after_identical=True)
        q['compiled']=details
        old_text=pins['original_TextModel_object'];assert base['all_linked_object_SHA256'][old_text['path']]==H(old_text['path'])==old_text['SHA256']
        old_pair=str(Path(base['native']).parent/'pair_probe.o')
        replaced={old_pair:objects['pair_probe'],old_text['path']:objects['text_model']}
        link=next(c[:]for c in base['commands']if '-shared'in c)
        assert set(replaced)<=set(link)
        link=[replaced.get(v,v)for v in link];link.insert(link.index('-o'),objects['masked_embedding'])
        native=B/'libKimi_vocab_parallel_embedding_r461.so';link[link.index('-o')+1]=str(native)
        run(link,'private-Native-link',40)
        actual_objects=[v for v in link if v.endswith('.o')];assert len(actual_objects)==34
        reused={p:h for p,h in base['all_linked_object_SHA256'].items()if p not in replaced};assert len(reused)==31
        assert all(p in actual_objects and H(p)==h for p,h in reused.items())
        def strong(path,label):
            rows=run(['nm','-D','--defined-only',str(path)],label,15)
            return {r.split()[-1]for r in rows.splitlines()if len(r.split())>=3 and r.split()[-2]in ['T','D','B','R']}
        old=strong(base['native'],'parent-strong-symbols');new=strong(native,'new-strong-symbols');assert old<=new
        old_capi={n for n in old if n.startswith('modern_')};assert old_capi<=new
        required={'modern_kimi_engine_create_vocab_parallel_r461','modern_kimi_vocab_embedding_capabilities_r461','modern_kimi_vocab_embedding_info_r461','modern_kimi_masked_embedding_r461'};assert required<=new
        dynamic=run(['readelf','--wide','-d',str(native)],'Native-dynamic',10);assert 'SYMBOLIC'in dynamic
        ldd=run(['ldd','-r',str(native)],'Native-ldd-relocations-only',20);assert 'not found'not in ldd and 'undefined symbol'not in ldd
        q.update(passed=True,completed=True,native=str(native),native_SHA256=H(native),all_linked_object_SHA256={p:H(p)for p in actual_objects},unchanged_objects_SHA256=reused,
            old_strong_exports_retained=len(old),old_CAPI_retained=len(old_capi),new_CAPI=sorted(required),actual_dependency_library_SHA256=base['actual_dependency_library_SHA256'],
            qualification='CPU source-body recording logic/metadata, compile/link/MD/ELF only. No SDK execution, Native dlopen, checkpoint weight load or actual BF16 collective/Model qualification.')
    except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        q.update(elapsed_s=time.monotonic()-start,Torch_imported='torch'in sys.modules);save()
        print(json.dumps({k:q.get(k)for k in ['actual_CPU','completed','passed','error','elapsed_s','native_SHA256']}),flush=True)
    return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
