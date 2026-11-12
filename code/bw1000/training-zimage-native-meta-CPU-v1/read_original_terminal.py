"""Independent stdlib original terminal/owners/source/meta inventories; no replay."""
from pathlib import Path
import datetime,hashlib,json,math,os,socket,sys
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-zimage-native-meta-CPU-v1')
files={}
def read(name):
    p=W/name;assert p.is_file() and 0<p.stat().st_size<=2*1024**2
    raw=p.read_bytes();files[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
    return json.loads(raw)
intent=read('intent.json');assert intent['claim']==CLAIM and intent['request_id']==REQUEST
out=read('outcome.json');cpu=read('CPU-result.json')
assert out['claim']==CLAIM and out['request_id']==REQUEST and out['CPU_result']==cpu
assert out['CPU_result_SHA256']==files['CPU-result.json']['SHA256']
assert out['owner']['host']==socket.gethostname() and out['epoch']==intent['epoch'] and out['deadline']==intent['deadline']
assert intent['deadline']-intent['epoch']==300 and cpu['epoch']==intent['epoch'] and cpu['deadline']==intent['epoch']+260
assert out['owned_group_reaped'] and out['wall_seconds']<300
def process(pid):
    try:
        f=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split()
        return dict(PID=pid,state=f[0],PGID=int(f[2]),startticks=int(f[19]))
    except (FileNotFoundError,ProcessLookupError):return None
for owner in (out['owner'],out['child'],cpu['owner']):
    current=process(owner['PID'])
    assert current is None or current['startticks']!=owner['startticks'] or current['state']=='Z'
for name in os.listdir('/proc'):
    if not name.isdigit():continue
    p=process(int(name))
    assert not p or p['PGID']!=out['child']['PGID'] or p['state']=='Z'
fm=read('src/FRAMEWORK_FILES.json');assert fm['framework_commit']=='8b43976c3f98602391a7f03674092a05ada2df60' and len(fm['files'])==356
for n,pin in fm['files'].items():
    p=W/'framework'/n;assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256'] and p.stat().st_mode&0o777==pin['mode']
binding=read('src/NATIVE_BINDING.json');root=Path(binding['source_root'])
for n,pin in binding['original_source_files'].items():
    p=root/n;assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
result=dict(original_terminal_consumed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    claim=CLAIM,request_id=REQUEST,original_returncode=out['returncode'],original_supervisor_passed=out['passed'],
    original_CPU_passed=cpu['passed'],owned_processes_absent=True,framework_files_verified=356,
    original_epoch=intent['epoch'],original_deadline=intent['deadline'],CPU_result=cpu,
    new_CPU_invocations=0,new_GPU_invocations=0,GPU_GO=False,Model_GO=False)
if cpu['passed']:
    assert out['passed'] and out['returncode']==0 and out['stop_reason'] is None and not cpu['GPU_initialized'] and not cpu['real_weight_or_data_open_attempts']
    headers=read('src/ORIGINAL_HEADERS.json');base={};adapter=None
    for n,v in headers.items():
        if n.startswith('diffusion_pytorch_model-'):
            assert not set(base)&set(v['tensors']);base.update(v['tensors'])
        else:adapter=v['tensors']
    assert len(base)==521 and len(adapter)==480
    original=read('BASE_METADATA.json');assert set(original)==set(base)
    for n,v in original.items():assert v==dict(shape=base[n]['shape'],dtype='torch.float32',meta=True)
    preset=read('PRESET_MODULE_METADATA.json')
    expected_preset={n[len('diffusion_model.'):-len('.lora_A.weight')] for n in adapter if n.endswith('.lora_A.weight')}
    assert set(preset)==expected_preset and len(preset)==240
    for n,v in preset.items():assert v==dict(module_class='torch.nn.modules.linear.Linear',shape=base[n+'.weight']['shape'],dtype='torch.float32',meta=True)
    targets={n[:-7]:v['shape'] for n,v in base.items() if n.endswith('.weight') and any(n[:-7].endswith('.'+s) for s in binding['target_modules'].split(','))}
    assert len(targets)==238
    trainable=read('TRAINABLE_METADATA.json');expected_trainable={}
    for n,shape in targets.items():
        expected_trainable[n+'.lora_A.default.weight']=[4,shape[1]]
        expected_trainable[n+'.lora_B.default.weight']=[shape[0],4]
    assert set(trainable)==set(expected_trainable) and sum(math.prod(v['shape']) for v in trainable.values())==9922560
    for n,v in trainable.items():assert v==dict(shape=expected_trainable[n],dtype='torch.bfloat16',meta=True,requires_grad=True)
    frozen=read('FROZEN_METADATA.json');expected_frozen={}
    for n,v in base.items():
        module,_,suffix=n.rpartition('.')
        name=module+'.base_layer.'+suffix if module in targets else n
        expected_frozen[name]=v['shape']
    assert set(frozen)==set(expected_frozen) and len(frozen)==521
    for n,v in frozen.items():assert v==dict(shape=expected_frozen[n],dtype='torch.bfloat16',meta=True,requires_grad=False)
    providers=read('IMPORTED_SOURCE_FILES.json')
    for n,pin in providers.items():
        p=root/pin['relative_path'];assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
    for key,name in [('base_metadata_SHA256','BASE_METADATA.json'),('preset_module_metadata_SHA256','PRESET_MODULE_METADATA.json'),
                     ('trainable_metadata_SHA256','TRAINABLE_METADATA.json'),('frozen_metadata_SHA256','FROZEN_METADATA.json'),
                     ('source_provider_manifest_SHA256','IMPORTED_SOURCE_FILES.json')]:assert cpu[key]==files[name]['SHA256']
    result.update(native_meta_qualified=True,actual_base_metadata_rows=521,actual_preset_module_rows=240,
        actual_trainable_rows=476,actual_trainable_elements=9922560,actual_frozen_rows=521,actual_provider_files=len(providers))
else:
    assert out['returncode']!=0 and not out['passed']
    result.update(native_meta_qualified=False,failure=read('failure-private.json'),
        planned_base_rows=521,planned_preset_module_rows=240,planned_trainable_rows=476)
result['original_files']=files
assert 'torch' not in sys.modules
print(json.dumps(result))
