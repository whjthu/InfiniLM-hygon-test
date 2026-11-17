"""One physicalGPU, one realTP1 component, scoped devicevalidation only."""
import ctypes as C,json,os,re,time
from pathlib import Path
from common import W,D,GPU_NODE,Budget,need,load,save,owner,qualify_cpu,file_pin,H
from assigned_kfd import assigned_kfd
def main():
    started=time.monotonic();intent=load(W/'install-intent.json');budget=Budget(intent['epoch']);budget.left(420)
    q=dict(passed=False,GPU_GO=False,Model_GO=False,device_validation_only=True,planned_pairs=3,planned_warm_pairs=2,planned_handle_rejections=24,planned_input_guards=10,planned_redzone_detectors=6,epoch=intent['epoch'],phase='admission')
    save(W/'worker-owner.json',owner(os.getpid()),True)
    try:
        claim=load(D/'CLAIM.json');need(claim['GPU_GO'] and claim['device_validation_GO'] and not claim['Model_GO'],'DEVICE_VALIDATION_NOT_APPROVED')
        mf=load(D/'SOURCE_FILES.json')
        for n,pin in mf.items():file_pin(D/n,pin,512*1024)
        e,build=qualify_cpu(host=False);gate=load(W/('gate-'+GPU_NODE+'.json'));step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];need(gate['step']==step and gate['kfd_holder_count']==0 and gate['boot_id']==owner(os.getpid())['boot_id'],'ACTUAL_GPU_GATE')
        need(0<=time.monotonic_ns()-gate['monotonic_ns']<=90_000_000_000,'GPU_GATE_STALE');q['step']=step
        cudart=Path('/opt/dtk/cuda/cuda/lib64/libcudart.so.12');pin=load(D/'RUNTIME_PINS.json')['cudart'];file_pin(cudart,pin,32*1024**2)
        rt=C.CDLL(str(cudart),mode=C.RTLD_LOCAL);count=C.c_int();rt.cudaGetDeviceCount.argtypes=[C.POINTER(C.c_int)];need(rt.cudaGetDeviceCount(C.byref(count))==0 and count.value==1,'ONE_VISIBLE_PHYSICAL_GPU')
        rt.cudaDeviceGetPCIBusId.argtypes=[C.c_char_p,C.c_int,C.c_int];buf=C.create_string_buffer(64);need(rt.cudaDeviceGetPCIBusId(buf,64,0)==0,'ASSIGNED_PCI');pci=buf.value.decode().lower()
        if re.fullmatch(r'[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',pci):pci='0000:'+pci
        need(pci in gate['pci'],'PCI_IN_FRESH_NODE');q['assigned_device']=assigned_kfd(pci)
        # Directprovider identity before factory: all9 publicfunctions resolve to
        # the exactqualified wrapper; selected-runtime mapping afterload recorded.
        lib=C.CDLL(e['DSO']['path'],mode=C.RTLD_LOCAL)
        class DlInfo(C.Structure):_fields_=[('name',C.c_char_p),('base',C.c_void_p),('symbol',C.c_char_p),('address',C.c_void_p)]
        dl=C.CDLL('libdl.so.2');dl.dladdr.argtypes=[C.c_void_p,C.POINTER(DlInfo)];dl.dladdr.restype=C.c_int
        providers={}
        for name in build['library']['strong_CABI']:
            info=DlInfo();need(dl.dladdr(C.cast(getattr(lib,name),C.c_void_p),C.byref(info)) and info.name,'CABI_PROVIDER');p=Path(info.name.decode()).resolve();need(p==Path(e['DSO']['path']).resolve(),'CABI_FOREIGN_PROVIDER');providers[name]=str(p)
        q['CABI_providers']=providers
        runtime_pins=load(D/'RUNTIME_PINS.json')['providers'];runtime_pins[e['parent_B']['path']]=e['parent_B']['SHA256'];selected={};total_provider_bytes=0
        symbols=[n for n in build['library']['imports'] if ('AttentionPrepare' in n and 'expand_cached' in n) or 'get_rank_info' in n or 'CommunicationGroupC' in n or n=='modern_glm_fp8_runtime_clear_r809']
        need(symbols and any('expand_cached_values' in n for n in symbols),'REAL_IMPORTS_SCOPE')
        for name in symbols:
            info=DlInfo();need(dl.dladdr(C.cast(getattr(lib,name),C.c_void_p),C.byref(info)) and info.name,'RUNTIME_PROVIDER')
            logical=Path(info.name.decode());canonical=logical.resolve();matches=[p for p in runtime_pins if Path(p).name==logical.name and Path(p).resolve()==canonical];need(matches and len({runtime_pins[p] for p in matches})==1,'PROVIDER_NOT_IN_ORIGINAL_CLOSURE')
            key=str(canonical)
            if key not in selected:
                raw=file_pin(canonical,dict(SHA256=runtime_pins[matches[0]],bytes=None),128*1024**2);total_provider_bytes+=len(raw);need(total_provider_bytes<=256*1024**2,'SELECTED_RUNTIME_PROVIDER_BOUND');selected[key]=dict(SHA256=H(raw),bytes=len(raw),symbols=[])
            selected[key]['symbols'].append(name)
        q['selected_Runtime_providers']=selected
        manifest=load(D/'PARAMETER_BINDING_TEMPLATE.json');manifest.update(GPU_GO=True,device_validation_scope=True,component_CPU_qualified=True,checkpoint_ranges_qualified=True,component_dso=e['DSO']['path'],component_dso_SHA256=e['DSO']['SHA256'])
        budget.left(420);q['phase']='load_pairs_guards_close';before=time.monotonic()
        from binding import run_component
        component=run_component(manifest,load(D/'CONTROLLED_INPUT_PINS.json'));q['component']=component;q['component_caller_wall_seconds']=time.monotonic()-before
        budget.left(420);need(component['passed'] and component['normal_close'] and component['completed_pairs']==3 and component['completed_warm_pairs']==2 and component['completed_handle_rejections']==24 and component['guards']['actual_rejections']==10 and component['guards']['redzone_detector_detected_restored']==6 and component['close']['registry_live_contexts']==0,'COMPLETE_DEVICE_VALIDATION')
        q.update(passed=True,phase='complete',component_numerical_qualified=True,Model_qualified=False,graph_exercised=False,global_Runtime_buffer_counter=None)
    except BaseException as error:
        import traceback
        private=dict(error_type=type(error).__name__,message=str(error)[:4096],traceback=traceback.format_exc()[:8192],phase=q['phase']);save(W/'failure-private.json',private,True)
        q.update(error_type=type(error).__name__,error_message_SHA256=H(str(error).encode()),phase_failed=q['phase'],private_failure_SHA256=H((W/'failure-private.json').read_bytes()))
    finally:q['full_worker_wall_seconds']=time.monotonic()-started;save(W/'device-result.json',q,True)
    return 0 if q['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
