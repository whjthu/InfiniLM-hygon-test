"""Changed actual compare/commit entry, recording TorchGPU/Native/16peer leaves.

Run only after a separate CPU claim/publication. No old suite or real SDK.
"""
import argparse,ctypes as C,hashlib,importlib,importlib.util,json,sys,tempfile,types
from pathlib import Path
import numpy as np

class Tensor(np.ndarray):
    def cpu(self):return self
    def float(self):return self.astype(np.float32)
    def abs(self):return np.abs(self)
    def square(self):return np.square(self)
    def sqrt(self):return np.sqrt(self)

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-provider',required=True);a=p.parse_args()
    d=Path(__file__).resolve().parent;parent=Path(a.parent_provider)/'public_glm_native'
    pkg=types.ModuleType('recorded_retention');pkg.__path__=[str(d),str(parent)];sys.modules[pkg.__name__]=pkg
    entry=importlib.import_module(pkg.__name__+'.accepted_prefix_continuation')
    def tensor(v,**kw):return np.array(v,dtype=kw['dtype']).view(Tensor)
    torch=types.SimpleNamespace(tensor=tensor,arange=lambda start,end,**kw:tensor(range(start,end),**kw),full=lambda shape,fill,**kw:tensor(np.full(shape,fill),**kw),float32=np.float32,int16=np.int16,int64=np.int64,bfloat16=np.float32,isfinite=np.isfinite,cuda=types.SimpleNamespace(synchronize=lambda r:None))
    resident=importlib.import_module(pkg.__name__+'.resident_continuation')
    above=dict(geometry='above',prompt=[10]*2949,proposal=list(range(20,28)),accepted=8)
    assert resident.select_above([dict(geometry='below'),above,dict(geometry='crossing_prompt_slice')])==[above]
    try:resident.select_above([dict(above,accepted=7)])
    except RuntimeError:pass
    else:raise AssertionError('actual resident selector accepted foreign geometry')
    out=[]
    for broken in [False,True]:
        calls=[];retained=[];state=dict(past=0);ctx=0x12345678abcdef01
        class F:
            def __init__(self,fn):self.fn=fn
            def __call__(self,*args):return self.fn(*args)
        def reset(pointer):assert pointer==ctx;state['past']=0;calls.append('reset');return 0
        def forward(pointer,width,count,desc,ids,pos,output,all_positions,stream):
            assert pointer==ctx and count==1 and desc[1]==state['past'];past=int(desc[1]);state['past']+=width
            rows=width if all_positions else 1
            words=(C.c_float*(rows*154880)).from_address(output)
            # Recording output, never used as model numerical evidence.
            np.ctypeslib.as_array(words).fill(1.)
            calls.append(('forward',past,int(width)));return 0
        def stage(pointer,sequence,past,width,ids,pos,output,stream):
            return forward(pointer,width,1,(0,past,width),ids,pos,output,1,stream)
        def prepare(pointer,seq,past,width,accepted):assert pointer==ctx and state['past']==past+width;calls.append('prepare');return 0
        def publish(pointer,seq,past,width,accepted):assert pointer==ctx;state['past']=past+accepted;calls.append('publish');return 0
        lib=types.SimpleNamespace(modern_glm_model_reset=F(reset),modern_glm_model_forward=F(forward),modern_glm_block_stage_v1=F(stage),modern_glm_block_prepare_v1=F(prepare),modern_glm_block_publish_v1=F(publish))
        def retain(label,output,metadata):
            if broken and label=='block-proposal':raise OSError('recorded retention failure')
            assert output.shape==(8 if label=='block-proposal'else 1,154880)
            retained.append(label);return dict(label=label,recording=True)
        error=None;value=None
        try:
            value=entry.compare_continuation(lib=lib,ctx=ctx,torch=torch,stage=1,rank=0,control=lambda v:v.__imul__(16),ptr=lambda v:int(v.ctypes.data),stream=lambda:0,sequence=2,prompt=tuple([10]*2949),chunk_widths=(512,512,512,512,512,389),proposal=tuple(range(20,28)),accepted=8,continuation=(28,29),retain=retain)
        except RuntimeError as exc:error=repr(exc)
        if broken:
            assert error and 'publish'not in calls and retained==['serial-proposal-%d'%i for i in range(8)]+['serial-continuation-0','serial-continuation-1']
        else:
            assert error is None and value['final_reset']and value['block']['length_evidence']['dynamic_per_slot_readback']=='UNKNOWN'
            assert len(retained)==13 and retained[10]=='block-proposal'and calls.count('reset')==3
            assert sum(isinstance(c,tuple)for c in calls)==25
        out.append(dict(passed=True,retained_labels=retained,expected_failure=error,model_calls=sum(isinstance(c,tuple)for c in calls),publish_calls=calls.count('publish')))
    retain_module=importlib.import_module(pkg.__name__+'.retain_logits');protocol=importlib.import_module(pkg.__name__+'.protocol')
    with tempfile.TemporaryDirectory()as tmp:
        run=protocol.Run(dict(run='recording-cpu',runtime_directory=tmp,actual_step='CPU-recording',source_manifest_SHA256='0'*64,provider_config_SHA256='1'*64,gpu_start_epoch=1.,Native_seconds=300))
        bits=np.full((8,154880),0x3f80,np.uint16)
        class Raw:
            dtype=torch.bfloat16;shape=bits.shape
            def detach(self):return self
            def cpu(self):return self
            def contiguous(self):return self
            def view(self,dtype):return self
            def numpy(self):return bits.view(np.int16)
        value=retain_module.make_retainer(run,torch,1,0)('block-proposal',Raw(),dict(recording=True))
        raw=Path(value['path']).read_bytes();assert raw==bits.tobytes()and hashlib.sha256(raw).hexdigest()==value['SHA256']and value['bytes']==2478080
        assert run.read('retained-1-0-block-proposal.json')['shape']==[8,154880]
        assert retain_module.make_retainer(run,torch,0,0)('block-proposal',None,{})is None
        schema=dict(passed=True,actual_atomic_raw_write_and_Run_identity=True,bytes=value['bytes'],SHA256=value['SHA256'],BF16_CPU_recording_words=True,actual_TorchGPU=False)
    return dict(passed=True,GPU=0,Native_execution=False,SDK_execution=False,actual_entry=True,actual_resident_selector=True,recording_leaves=['Torch device/tensors represented in CPU float32, not BF16 numerical qualification','Native functions','16-owner control reduction','synthetic same geometry IDs; device must use exact private above plan'],cases=out,raw_schema=schema,old_suites_reexecuted=False)
if __name__=='__main__':print(json.dumps(main(),indent=2))
