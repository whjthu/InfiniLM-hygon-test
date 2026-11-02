"""Actual ctypes adapter/body consumed with recording Native/CUDA leaves."""
import base64,ctypes as C,json
from types import SimpleNamespace
from constructor import body
from public_api import Device,declare,SYMBOLS

CFG=dict(num_hidden_layers=78,hidden_size=6144,indexer_types=['full']*78)
PAYLOAD=dict(world=base64.b64encode(b'world16').decode(),tp=[base64.b64encode(b'tp8a').decode(),base64.b64encode(b'tp8b').decode()],control=base64.b64encode(b'control16').decode())

class Fn:
    def __init__(self,call):self.call=call
    def __call__(self,*a):return self.call(*a)

class Lib:
    def __init__(self,stage,rank,fault=None):
        self.stage,self.rank,self.fault=stage,rank,fault;self.events=[];self.created=False;self.destroyed=False
        for name in SYMBOLS:setattr(self,name,Fn(lambda *a,n=name:self.call(n,*a)))
    def call(self,name,*a):
        self.events.append(name)
        if name=='modern_glm_model_error':return b'recording leaf error'
        if name=='modern_glm_model_create':
            assert list(a[:4])==[self.rank,self.stage,1,32768]
            assert json.loads(a[4])==CFG
            assert a[5].raw.rstrip(b'\0')==b'world16' and a[6].raw.rstrip(b'\0')==[b'tp8a',b'tp8b'][self.stage]
            self.created=self.fault!='create';return 12345 if self.created else None
        if name=='modern_glm_partition_report_r3854':
            return json.dumps(dict(PP=2,stage=self.stage,device=self.rank,begin=[0,38][self.stage],end=[38,78][self.stage],weights_ready=False,keys=['unloaded.test.weight'])).encode()
        if name=='modern_glm_constructor_census_r4158':
            return json.dumps(dict(layer_begin=[0,38][self.stage],layer_end=[38,78][self.stage],rows=1,capacity=32768,loaded=False,
                    cache_category_bytes={'latent':1024,'projected_key':2048,'projected_value':2048},cache_unique_start_bytes=5120,weight_unique_start_bytes=100000,weight_keys=1,
                    scope='CPU recording logical bytes; not actual Native/GPU allocation')).encode()
        if name=='modern_glm_unloaded_trim_r4052':
            assert a[2]==6
            for i,x in enumerate([600,700,800,self.rank,self.stage,2]):a[1][i]=x
            return int(self.fault=='trim')
        if name=='modern_owner_affinity_report_r2392':
            return json.dumps(dict(owners=[dict(device=d,changes=int(d==self.rank),verified_mask=','.join(map(str,range(d*4,d*4+4))) if d==self.rank else '') for d in range(8)])).encode()
        if name=='modern_glm_model_control':return 0
        if name=='modern_glm_model_destroy':self.destroyed=True;return int(self.fault=='destroy')
        if name=='modern_glm_model_shutdown':return int(self.fault=='shutdown')
        raise AssertionError('Unexpected public ABI call '+name)

class Torch:
    float32='recordingfloat32'
    def __init__(self,lib):
        self.lib=lib
        self.cuda=SimpleNamespace(synchronize=lambda rank:None,mem_get_info=lambda rank:(59<<30,68702699520),
                memory_allocated=lambda rank:0,memory_reserved=lambda rank:0,current_stream=lambda:SimpleNamespace(cuda_stream=1234))
    def ones(self,*a,**kw):return SimpleNamespace(data_ptr=lambda:5678,item=lambda:15 if self.lib.fault=='world' else 16)

def case(stage,rank,fault=None):
    lib=declare(Lib(stage,rank,fault));deps=Device(lib,Torch(lib),rank,affinity_reader=lambda r:list(range(r*4,r*4+4)));saved=[];error=None
    def gate(label):
        if fault=='cancel' and label=='before-create':raise RuntimeError('recordingcancel')
        if fault=='timeout' and label=='after-create':raise TimeoutError('recordingdeadline')
    try:q=body(deps,stage,rank,CFG,PAYLOAD,gate,lambda q:saved.append(json.loads(json.dumps(q))))
    except BaseException as e:error=e;q=saved[-1]
    return q,lib,error

def run():
    cases=[];owners=[]
    for stage in range(2):
        for rank in range(8):
            q,lib,error=case(stage,rank)
            assert error is None and q['completed'] and q['normal_destroy'] and q['normal_shutdown']
            assert q['loaded_service_admission_passed'] is False and not q['Model_qualified']
            assert lib.events.count('modern_glm_model_create')==lib.events.count('modern_glm_model_destroy')==lib.events.count('modern_glm_model_shutdown')==1
            assert q['memory_accounting']['Native_allocator_live_bytes'] is None
            owners.append(dict(stage=stage,rank=rank,global_rank=q['global_rank'],affinity=q['owner_affinity']['CPUs']))
    assert {o['global_rank'] for o in owners}==set(range(16))
    for stage in range(2):assert len({cpu for o in owners if o['stage']==stage for cpu in o['affinity']})==32
    cases.append('ActualDevice ctypes binding/create argv/world16/twoTP8 payload and completebody16owner census/trim/normaldestroy;Native/CUDA leaves recorded')
    for fault in ['create','trim','world','destroy','shutdown','cancel','timeout']:
        q,lib,error=case(1,7,fault);assert error is not None and not q['completed']
        assert lib.events.count('modern_glm_model_create')<=1
        if lib.created:assert lib.destroyed
        assert lib.events.count('modern_glm_model_shutdown')==1
        if fault=='destroy':assert not q['normal_destroy']
        if fault=='cancel':assert not lib.created
    cases.append('Create/null,trim,world16,deadline,cancel,destroy,shutdown failure saves partialreport and always attempts ownedcleanup;no retries')
    return dict(completed=True,passed=True,GPU=0,Native_dlopen=False,weights=False,owners=owners,cases=cases,
                actual_native_group=False,actual_allocator_qualification=False)

if __name__=='__main__':print(json.dumps(run(),indent=2))

def check_go(config_sha):
    """Actual future entry admission, real configmetadata, recorded GO/receipt."""
    from pathlib import Path
    import hashlib,os,tempfile
    from unittest import mock
    import GPU_entry as entry
    checked=[]
    for fault in [None,'source','CPU','Model','deadline','node']:
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);directory=root/'source';directory.mkdir();(root/'control').mkdir();out=root/'candidates/GLM-C1-PP2-constructor-r395';out.mkdir(parents=True)
            (directory/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(files_SHA256={})))
            cpu=out/'CPU-result.json';cpu.write_text(json.dumps(dict(passed=fault!='CPU',completed=True,GPU=0,Native_dlopen=False)))
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            go=dict(GPU_GO=True,operation='constructor_only_no_weights_finish_forward',source_manifest_SHA256=sha(directory/'SOURCE_MANIFEST.json'),CPU_receipt_SHA256=sha(cpu),
                    Native_SHA256=entry.NATIVE_SHA,TP=8,PP=2,rows=1,capacity=32768,GPU_max_s=300,Model_GO=fault=='Model',original_GPU_start_unix=1000.,publication_commit='1'*40,config_SHA256=config_sha)
            if fault=='source':go['source_manifest_SHA256']='0'*64
            path=root/'control/go.json';path.write_text(json.dumps(go))
            env=dict(GLM_R395_GO=str(path),GLM_R395_DEADLINE_UNIX='1301' if fault=='deadline' else '1300',SLURM_JOB_ID='910376',SLURM_PROCID='0',GLM_PP_SIZE_R3603='2')
            error=None
            with mock.patch.object(entry,'R',root),mock.patch.object(entry,'D',directory),mock.patch.dict(os.environ,env),mock.patch.object(entry.socket,'gethostname',return_value='wrongnode' if fault=='node' else 'a02r4n12'):
                try:entry.read_go()
                except (AssertionError,ValueError) as e:error=e
            assert (error is None)==(fault is None)
            checked.append(fault or 'exact_recorded_GO_realconfig')
    return dict(passed=True,cases=checked,actual_GPU=False,actual_Native_dlopen=False,GO_receipt_leaves_recorded=True)
