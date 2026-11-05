"""Bounded CPU-only two-HOST build. Does not dlopen or execute a Native entry."""
import argparse,ast,ctypes,hashlib,json,os,re,subprocess,sys,time,runpy
from pathlib import Path
DEST=Path('/refactor/candidates/GLM35c-layer0-checkpoint-component-CPU-v1')
PARENT=Path('/refactor/candidates/GLM952-block-prefix-native-v1')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def parse_md(raw):return raw.replace('\\\n',' ').split(':',1)[1].split()
def run(package,claim_path,claim_sha):
    claim_raw=claim_path.read_bytes();assert hashlib.sha256(claim_raw).hexdigest()==claim_sha
    claim=json.loads(claim_raw);step=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','')
    assert claim['CPU_GO']is True and claim['GPU_GO']is False and claim['Model_GO']is False
    assert claim['parent_job']==os.environ['SLURM_JOB_ID'] and re.fullmatch(r'\d+\.\d+',step)
    assert os.environ['GLM_L0_CPU_CLAIM_SHA']==claim_sha and claim['destination']==str(DEST)
    assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
    assert not any(os.environ.get(k)for k in ['LD_PRELOAD','LD_AUDIT','GCC_EXEC_PREFIX','COMPILER_PATH','LIBRARY_PATH','CPATH','CPLUS_INCLUDE_PATH','C_INCLUDE_PATH'])
    mf_raw=(package/'SOURCE_MANIFEST.json').read_bytes();assert hashlib.sha256(mf_raw).hexdigest()==claim['source_manifest_SHA256'];mf=json.loads(mf_raw)
    for name,h in mf['files_SHA256'].items():assert digest(package/name)==h,(name,'source SHA')
    for name in ['derive_caller.py','component_api.py','build_CPU.py']:ast.parse((package/name).read_text())
    pins=json.loads((package/'PARENT_PINS.json').read_bytes());der=json.loads((package/'DERIVATION.json').read_bytes())
    assert digest(PARENT/'libGLM952_block_prefix_v1.so')==pins['Native_SHA256']
    assert digest(PARENT/'CPU-build.json')==pins['CPU835_receipt_SHA256']
    for p,v in pins['objects'].items():assert digest(p)==v['SHA256'],p
    for p,v in pins['parent_MD'].items():assert digest(p)==v['SHA256'],p
    tools=json.loads((package/'TOOLCHAIN.json').read_bytes())['tools']
    for p,v in tools.items():assert str(Path(p).resolve())==v['canonical']and digest(p)==v['sha256'],p
    DEST.mkdir(exist_ok=False);src=DEST/'src';src.mkdir();deadline=time.monotonic()+claim['inside_seconds']
    report=dict(completed=False,CPU=True,actual_CPU=step,GPU=False,Model=False,Native_execution=False,dlopen=False,source_manifest_SHA256=claim['source_manifest_SHA256'],claim_SHA256=claim_sha,phase='derive',retained_objects=28,new_HOST=2,new_HIP=0)
    def command(name,argv):
        left=deadline-time.monotonic()-4
        if left<=0:raise TimeoutError('bounded CPU deadline')
        with (DEST/(name+'.stdout')).open('xb')as out,(DEST/(name+'.stderr')).open('xb')as err:
            p=subprocess.run(['/usr/bin/timeout','--kill-after=3s',str(left)+'s',*argv],stdout=out,stderr=err)
        if p.returncode:raise RuntimeError(name+' rc='+str(p.returncode))
        return (DEST/(name+'.stdout')).read_text()
    try:
        derive=runpy.run_path(str(package/'derive_caller.py'))
        oldsource=Path(pins['caller_argv'][pins['caller_argv'].index('-c')+1]);caller=derive['caller'](oldsource.read_bytes());selected=derive['selected']((oldsource.parent/'selected_values_r400.hpp').read_bytes())
        assert hashlib.sha256(caller).hexdigest()==der['derived_caller_SHA256']and hashlib.sha256(selected).hexdigest()==der['derived_selected_header_SHA256']
        (src/'caller.cpp').write_bytes(caller);(src/'selected_values_r400.hpp').write_bytes(selected)
        expected=dict(pins['dependency_SHA256']);expected.update({str(package/n):h for n,h in mf['files_SHA256'].items()});expected.update({str(src/'caller.cpp'):der['derived_caller_SHA256'],str(src/'selected_values_r400.hpp'):der['derived_selected_header_SHA256']})
        original=pins['caller_argv'];common=original[:original.index('-MD')]
        common=[common[0],'-I'+str(src),'-I'+str(package),*common[1:]]
        counts=[];outputs=[]
        report['phase']='preprocess_compile'
        for i,source in enumerate([src/'caller.cpp',package/'layer0_component.cpp']):
            output=DEST/('new'+str(i)+'.o');md=DEST/('new'+str(i)+'.d');outputs.append(str(output))
            def verify(path):
                rows=parse_md(path.read_text());missing=[]
                for p in rows:
                    if p not in expected:missing.append(p)
                    elif digest(p)!=expected[p]:raise ValueError('dependency SHA differs '+p)
                if missing:raise ValueError('unpinned fresh MD dependencies '+repr(missing))
                return len(rows)
            command('preprocess'+str(i),[*common,'-M','-MF',str(md),'-MT',str(output),str(source)]);verify(md)
            command('compile'+str(i),[*common,'-MD','-MF',str(md),'-c',str(source),'-o',str(output)]);counts.append(verify(md))
        report['phase']='link'
        oldcaller=str(PARENT/'obj/07.o');argv=[outputs[0]if p==oldcaller else p for p in pins['link_argv']];assert argv.count(outputs[0])==1
        argv.insert(argv.index('-o'),outputs[1]);library=DEST/'libGLM35c_layer0_component.so';argv[argv.index('-o')+1]=str(library);command('link',argv)
        def exports(text):
            result={}
            for row in text.splitlines():
                _,kind,name=row.split()
                if kind in {'W','V','u','w','v'}and name.startswith('_Z'):continue
                result[name]=kind
            return result
        report['phase']='ELF_and_typed_CPU'
        before=exports(command('parent-nm',['/usr/bin/nm','-D','--defined-only',str(PARENT/'libGLM952_block_prefix_v1.so')]))
        after=exports(command('candidate-nm',['/usr/bin/nm','-D','--defined-only',str(library)]))
        added={'glm_l0_'+x for x in ['error','id_size','id','create','keys','load','finish','step','snapshot','read','reset','destroy']}
        assert all(after.get(k)==v for k,v in before.items())and set(after)-set(before)==added
        dyn=lambda s:re.findall(r'\((NEEDED|RPATH|RUNPATH|SONAME)\).*?\[([^\]]+)\]',s)
        olddyn=dyn(command('parent-readelf',['/usr/bin/readelf','-d',str(PARENT/'libGLM952_block_prefix_v1.so')]))
        assert olddyn==dyn(command('candidate-readelf',['/usr/bin/readelf','-d',str(library)]))
        command('ldd-r',['/usr/bin/ldd','-r',str(library)])
        api=runpy.run_path(str(package/'component_api.py'));ctx=0x1234567800000042;seen=[]
        # Real libffi pointer width; no SDK or Native callable is loaded/executed.
        class Leaf:pass
        leaf=Leaf();funcs=[]
        specs={'error':([],ctypes.c_char_p),'id_size':([],ctypes.c_size_t),'id':([ctypes.c_void_p],ctypes.c_int),'create':([ctypes.c_int,ctypes.c_int,ctypes.c_char_p,ctypes.c_void_p,ctypes.c_int],ctypes.c_void_p),'keys':([ctypes.c_void_p],ctypes.c_char_p),'load':([ctypes.c_void_p,ctypes.c_char_p],ctypes.c_int),'finish':([ctypes.c_void_p],ctypes.c_int),'step':([ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_int],ctypes.c_int),'snapshot':([ctypes.c_void_p],ctypes.c_char_p),'read':([ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t],ctypes.c_int),'reset':([ctypes.c_void_p],ctypes.c_int),'destroy':([ctypes.c_void_p,ctypes.c_void_p],ctypes.c_int)}
        for name,(args,ret)in specs.items():
            def callback(*a,_name=name):
                if _name in {'finish','reset','load','step','read','destroy'}:seen.append((_name,a[0]));return 0
                return ctx if _name=='create'else 1 if _name in {'id','id_size'}else b'[]'
            f=ctypes.CFUNCTYPE(ret,*args)(callback);funcs.append(f);setattr(leaf,'glm_l0_'+name,f)
        api['declare'](leaf);assert leaf.glm_l0_create(1,0,b'{}',1,0)==ctx
        assert leaf.glm_l0_finish(ctx)==0 and leaf.glm_l0_reset(ctx)==0 and seen==[('finish',ctx),('reset',ctx)]
        report.update(completed=True,phase='completed',library=str(library),library_SHA256=digest(library),fresh_dependency_counts=counts,original_strong_exports=len(before),new_CABI=sorted(added),dynamic_metadata_equal=True,typed_high64_recording_passed=True,recording_Native_execution=False,objects_SHA256={p:digest(p)for p in outputs},device_qualification=False)
    except BaseException as e:report.update(error=type(e).__name__+': '+str(e));raise
    finally:(DEST/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('claim',type=Path);a.add_argument('claim_SHA256');x=a.parse_args();print(json.dumps(run(Path(__file__).resolve().parent,x.claim,x.claim_SHA256)))
