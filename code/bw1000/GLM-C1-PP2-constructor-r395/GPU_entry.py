"""Explicit future constructor-only GO. No GPU claim is issued by this source."""
from pathlib import Path
import base64,ctypes,hashlib,json,os,re,time,traceback,math,socket
from constructor import body,NATIVE_SHA
from public_api import Device,declare,providers,unique

D=Path(__file__).resolve().parent;R=Path('/refactor')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def gate(label):
    if time.time()>=float(os.environ['GLM_R395_DEADLINE_UNIX']):raise TimeoutError('Original constructorbound:'+label)
    if Path(os.environ['GLM_R395_CANCEL_FILE']).exists():raise RuntimeError('Original constructorcancel:'+label)

def read_go():
    path=Path(os.environ['GLM_R395_GO']);assert path.parent==R/'control'
    go=json.loads(path.read_bytes());assert go['GPU_GO'] is True and go['operation']=='constructor_only_no_weights_finish_forward'
    assert go['source_manifest_SHA256']==sha(D/'SOURCE_MANIFEST.json')
    assert go['CPU_receipt_SHA256']==sha(R/'candidates/GLM-C1-PP2-constructor-r395/CPU-result.json')
    cpu=json.loads((R/'candidates/GLM-C1-PP2-constructor-r395/CPU-result.json').read_bytes())
    assert cpu['passed'] and cpu['completed'] and cpu['GPU']==0 and cpu['Native_dlopen']is False
    assert go['Native_SHA256']==NATIVE_SHA and go['TP']==8 and go['PP']==2 and go['rows']==1 and go['capacity']==32768
    assert go['GPU_max_s']==300 and go['Model_GO'] is False
    epoch=go['original_GPU_start_unix'];assert math.isfinite(epoch) and epoch>0
    assert float(os.environ['GLM_R395_DEADLINE_UNIX'])==epoch+300
    assert re.fullmatch('[0-9a-f]{40}',go['publication_commit'])
    assert os.environ['SLURM_JOB_ID']=='910376' and os.environ['GLM_PP_SIZE_R3603']=='2'
    stage=int(os.environ['SLURM_PROCID']);assert stage in range(2)
    assert socket.gethostname().split('.')[0]==['a02r4n12','a02r4n13'][stage]
    for n,h in json.loads((D/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(D/n)==h,n
    config=Path('/old/models/ZhipuAI-GLM-5.3/config.json');assert sha(config)==go['config_SHA256']
    return go,json.loads(config.read_bytes())

def worker(rank,payload,native):
    go,config=read_go();gate('worker-before-GPU-import')
    assert native['library_SHA256']==NATIVE_SHA and sha(native['library_actual_path'])==NATIVE_SHA
    import torch
    torch.set_num_threads(1);torch.cuda.set_device(rank)
    stage=int(os.environ['SLURM_PROCID']);assert stage in range(2)
    prop=torch.cuda.get_device_properties(rank);assert 'gfx936' in prop.gcnArchName and prop.multi_processor_count==80
    lib=declare(ctypes.CDLL(native['library_actual_path']));mapping=providers(lib,native['library_actual_path'])
    path=R/'results'/('GLM-C1-PP2-constructor-r395-'+go['run']+'-stage%d-rank%d.json'%(stage,rank));assert not path.exists()
    def save(q):
        q.update(run=go['run'],actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
                 source_manifest_SHA256=go['source_manifest_SHA256'],CPU_receipt_SHA256=go['CPU_receipt_SHA256'],CAPI_providers=mapping,
                 actual_hardware=dict(name=prop.name,gcnArchName=prop.gcnArchName,total_memory=prop.total_memory,multi_processor_count=prop.multi_processor_count))
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2)+'\n');tmp.replace(path)
    body(Device(lib,torch,rank),stage,rank,config,payload,gate,save)

def main():
    gate('before-verified-imports');go,_=read_go();stage=int(os.environ['SLURM_PROCID']);assert stage in range(2)
    from native_chain_r186 import verify
    native=verify();assert native['library_SHA256']==NATIVE_SHA
    gate('before-Torch-parent-import')
    assert len(os.sched_getaffinity(0))>=32
    import torch
    import torch.multiprocessing as mp
    assert torch.cuda.device_count()==8
    path=R/'control'/('GLM-C1-PP2-constructor-r395-'+go['run']+'-ids.json')
    if stage==0:
        lib=declare(ctypes.CDLL(native['library_actual_path']))
        value=dict(world=unique(lib),tp=[unique(lib),unique(lib)],control=unique(lib))
        with path.open('x')as f:json.dump(value,f);f.write('\n')
    while True:
        gate('wait-original-IDs')
        try:
            raw=path.read_bytes()
            if raw.endswith(b'\n'):payload=json.loads(raw);break
        except FileNotFoundError:pass
        time.sleep(.1)
    mp.spawn(worker,args=(payload,native),nprocs=8,join=True)
    print('GLM_C1_PP2_R395_NORMAL '+str(stage),flush=True)

if __name__=='__main__':main()
