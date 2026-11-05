"""New caller ABI/ownership recording only. No CDLL/SDK/Runtime/device execution."""
import ctypes as C
import importlib.util
import json
import weakref
from pathlib import Path
D=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('typed470',D/'typed_binding.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class Recorder:
    def __init__(self):
        self.callbacks=[];self.buffers=[];self.calls=[];self.errors=[];self.ctx=0x123456789AB;self.rank=15;self.ready=False;self.loaded=False;self.closed=False
        for name,(args,ret)in m.SIGNATURES.items():
            def callback(*a,name=name):
                try:return self.call(name,a)
                except BaseException as error:self.errors.append(repr(error));return 0
            # char* callbacks return stable borrowed buffers just like the CABI.
            f=C.CFUNCTYPE(C.c_void_p if ret is C.c_char_p else ret,*args)(callback)
            setattr(self,'modern_glm_global_'+name+'_r470',f);self.callbacks.append(f)
    def text(self,q):
        b=C.create_string_buffer((q if isinstance(q,str)else json.dumps(q)).encode());self.buffers.append(b);return C.addressof(b)
    def call(self,name,a):
        self.calls.append(name)
        if name=='id_size':return 8
        if name=='error':return self.text('recording Native failure')
        if name=='create':
            assert a[:4]==(1,7,1,32768)and a[7]==8
            assert C.string_at(a[5],8)==b'worlduid'and C.string_at(a[6],8)==b'tensorid';assert json.loads(a[4])['glm_global_tp16_Model_r470'];return self.ctx
        assert a[0]==self.ctx,(name,a)
        if name=='keys':return self.text(['model.layers.77.mlp.experts.255.down_proj.weight'])
        if name=='load_batch':
            q=json.loads(a[1]);assert q[0]['shape']==[6144,2048]and q[0]['pointer']==0x123456789 and q[0]['dtype']==2;self.loaded=True;return 0
        if name=='finish':assert self.loaded;self.ready=True;return 0
        if name=='geometry':return self.text(dict(TP=16,PP=1,world_size=16,world_rank=15,tp_rank=15,rows=1,capacity_per_row=32768,local_attention_heads=4,local_index_heads=2,root_output_rank=0,Model_READY=self.ready))
        if name=='forward':assert self.ready and a[1:3]==(3,1)and list(a[3][:3])==[0,0,3]and a[4:7]==(0x100000001,0x200000002,0x300000003)and a[7:]==(0,0x400000004);return 0
        if name=='control':assert a[1:]==(1,0x500000005,0x400000004);return 0
        if name=='reset':return 0
        if name=='destroy':self.closed=True;return 0
        raise AssertionError(name)
class Source:pass

def run():
    lib=Recorder();cfg=dict(glm_global_tp16_r470=True,glm_global_tp16_Model_r470=True)
    ctx=m.GlobalTP16Context(lib,cfg,1,7,b'worlduid',b'tensorid');assert ctx.ctx==lib.ctx
    assert ctx.geometry()['Model_READY']is False and len(ctx.keys())==1
    source=Source();ref=weakref.ref(source)
    ctx.load_batch([dict(name=ctx.keys()[0],shape=[6144,2048],dtype=2,pointer=0x123456789,owner=source)]);assert ref()is source
    ctx.finish();assert ctx.geometry()['Model_READY']is True
    ctx.forward(3,0,0x100000001,0x200000002,0x300000003,0x400000004);ctx.control(1,0x500000005,0x400000004)
    for wrong in [(1,1),(0,3)]:
        try:ctx.forward(wrong[0],wrong[1],1,2,3,0)
        except ValueError:pass
        else:raise AssertionError('bad past/width accepted')
    ctx.reset();assert ctx.past==0;ctx.close();assert lib.closed and ctx.ctx is None
    try:m.GlobalTP16Context(lib,{},1,7,b'worlduid',b'tensorid')
    except ValueError:pass
    else:raise AssertionError('defaultOFF missing')
    assert not lib.errors,lib.errors
    return dict(passed=True,actual_new_typed_caller=True,high64_CABI=True,global_rank=15,local_device=7,global_unsharded_loader_descriptor=True,strong_source_until_load_returns=True,past_negative_cases=2,defaultOFF=True,normal_close_recording=True,Native_leaf='CFUNCTYPE recording only',Native_dlopen=False,SDK=False,Runtime=False,GPU=False,Model_READY=False,calls=lib.calls)
if __name__=='__main__':print(json.dumps(run(),indent=2))
