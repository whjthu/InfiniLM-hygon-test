"""Four-node AR-only constructor census. Metadata mode has no device imports."""
from pathlib import Path
import argparse,ctypes as C,hashlib,json,os,socket,time,traceback
from constructor import check_metadata,resident_constructor
D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def metadata(node=None,hostname=None):
    q=json.loads((D/'config.json').read_bytes())
    n=int(os.environ['SLURM_PROCID']) if node is None else node
    h=socket.gethostname().split('.')[0] if hostname is None else hostname
    assert n in range(4) and h==q['nodes'][n], (n,h)
    assert len(os.sched_getaffinity(0))>=32, 'preserve original entry mask; no early pin'
    for key,value in q['environment'].items(): assert os.environ.get(key)==value,(key,os.environ.get(key))
    root=Path(q['model_directory'])
    assert sha(root/'config.json')==q['config_SHA256']
    assert sha(root/'model.safetensors.index.json')==q['index_SHA256']
    cfg=json.loads((root/'config.json').read_bytes())
    wm=json.loads((root/'model.safetensors.index.json').read_bytes())['weight_map']
    check_metadata(cfg,wm)
    return q,n,cfg,wm

def main():
    p=argparse.ArgumentParser();p.add_argument('--metadata-only',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--metadata-all-recording-hosts',action='store_true');a=p.parse_args()
    assert sum([a.metadata_only,a.execute,a.metadata_all_recording_hosts])==1
    if a.metadata_all_recording_hosts:
        cfg=json.loads((D/'config.json').read_bytes());roles=[]
        for n,h in enumerate(cfg['nodes']):
            _,rank,_,_=metadata(n,h);roles.append(dict(node=rank,hostname_leaf='recording:'+h))
        print(json.dumps(dict(passed=True,roles=roles,Native_dlopen=False,GPU=0,real_affinity_count=len(os.sched_getaffinity(0)))));return
    q,n,cfg,wm=metadata()
    if a.metadata_only:
        print(json.dumps(dict(passed=True,node=n,metadata_only=True,Native_dlopen=False,GPU=0)));return
    epoch=float(os.environ['K459_EPOCH']);assert time.time()<epoch+q['Native_seconds']
    out=Path(q['runtime_directory']);out.mkdir(exist_ok=True)
    report=out/('node%d.json'%n);assert not report.exists()
    result=dict(node=n,hostname=q['nodes'][n],run=q['run'],source_manifest_SHA256=os.environ['K459_SOURCE_SHA'],
        epoch=epoch,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],PID=os.getpid(),UID=os.getuid(),
        PGID=os.getpgrp(),entry_mask=sorted(os.sched_getaffinity(0)),completed=False,passed=False,
        weights=0,forward=0,HTTP=0,reset=0,create_calls=0,destroy_calls=0,events=[])
    def save():
        t=report.with_suffix('.tmp');t.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');t.replace(report)
    def record(event,payload):
        result['events'].append(dict(event=event,payload=payload,unix=time.time()))
        if event=='before_create':result['create_calls']+=1
        if event=='destroy_returned':result['destroy_calls']+=1
        save()
    save()
    try:
        import torch
        assert torch.cuda.device_count()==8
        cuda=C.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so')
        cuda.cudaDeviceGetPCIBusId.argtypes=[C.c_char_p,C.c_int,C.c_int];cuda.cudaDeviceGetPCIBusId.restype=C.c_int
        hardware=[]
        for d in range(8):
            prop=torch.cuda.get_device_properties(d);bus=C.create_string_buffer(64)
            assert 'gfx936'in prop.gcnArchName and cuda.cudaDeviceGetPCIBusId(bus,64,d)==0
            hardware.append(dict(device=d,PCI=bus.value.decode().lower(),arch=prop.gcnArchName))
        assert len({v['PCI']for v in hardware})==8
        assert {v['PCI']for v in hardware}=={p.resolve().name for p in Path('/sys/class/drm').glob('renderD*/device')}
        result['hardware']=hardware
        torch.cuda.set_device(0)
        def memory():return [list(torch.cuda.mem_get_info(d))for d in range(8)]
        result['memory_before_create']=memory();save()
        assert all(t==q['physical_VRAM_bytes']and f>=q['floor_bytes']for f,t in result['memory_before_create'])
        assert sha(q['native'])==q['Native_SHA256']
        lib=C.CDLL(q['native'],mode=os.RTLD_NOW|os.RTLD_LOCAL)
        class Info(C.Structure):_fields_=[('file',C.c_char_p),('base',C.c_void_p),('symbol',C.c_char_p),('address',C.c_void_p)]
        dl=C.CDLL(None);dl.dladdr.argtypes=[C.c_void_p,C.POINTER(Info)];dl.dladdr.restype=C.c_int
        providers={}
        for name in ['modern_kimi_engine_create_ar_storage_r456','modern_kimi_ar_storage_capabilities_r456','modern_kimi_engine_destroy_r2475']:
            info=Info();assert dl.dladdr(C.cast(getattr(lib,name),C.c_void_p),C.byref(info))
            path=Path(info.file.decode()).resolve();assert path==Path(q['native']).resolve() and sha(path)==q['Native_SHA256']
            providers[name]=dict(path=str(path),SHA256=sha(path),address=int(info.address))
        maps=Path('/proc/self/maps').read_text();result.update(dladdr=providers,Native_maps=maps)
        for basename,pin in q['dependency_libraries'].items():
            matching={line.split()[-1]for line in maps.splitlines()if line.split()and line.split()[-1].split('/')[-1]==basename}
            assert len(matching)==1,(basename,matching)
            assert all(sha(v)==pin['SHA256']for v in matching),basename
        save()
        with resident_constructor(lib,cfg,wm,node=n,address=q['head'],port=q['native_port'],record=record):
            result['memory_after_create']=memory();save()
            assert all(t==q['physical_VRAM_bytes']and f>=q['floor_bytes']for f,t in result['memory_after_create']), 'original 4GiB constructor floor'
            assert time.time()<epoch+q['Native_seconds'],'original epoch Native deadline'
        result['memory_after_destroy']=memory()
        assert result['create_calls']==result['destroy_calls']==1
        assert not any(v['event']=='cleanup_error'for v in result['events'])
        result['passed']=True
    except BaseException as error:
        result.update(error=repr(error),traceback=traceback.format_exc());raise
    finally:
        result['completed']=True;result['finished_unix']=time.time();save()

if __name__=='__main__':main()
