"""Execute the new actual Python loader with recording checkpoint/Native leaves."""
from pathlib import Path
import gc,json,sys,tempfile,weakref
from types import SimpleNamespace
import head_source_r461 as head
from parameter_loader_r461 import load_parameters

class Fn:
    def __init__(self,f):self.f=f
    def __call__(self,*a):return self.f(*a)

def main():
    torch=SimpleNamespace(float32='f32',bfloat16='bf16',uint8='u8',uint16='u16')
    visited=[];live=weakref.WeakValueDictionary();serial=[0]
    class W:
        device=SimpleNamespace(type='cpu');dtype='bf16';shape=[163840,7168]
        def __init__(self,key):
            serial[0]+=1;self.pointer=0x100000+serial[0]*0x100;self.key=key;live[self.pointer]=self
        def is_contiguous(self):return True
        def contiguous(self):return self
        def numel(self):return 163840*7168
        def element_size(self):return 2
        def data_ptr(self):return self.pointer
        def view(self,dtype):assert dtype=='u16';return self
        def numpy(self):return self
        def __getitem__(self,row):
            visited.append(row);assert 0<=row<163840
            return (0x3f00+((row+column)%127)for column in range(7168))
    registered=[];loads=[];copy=[0]*5
    def register(payload):
        q=json.loads(payload);assert len(q['ranks'])==32
        assert [r['rank']for r in q['ranks']]==list(range(32))
        assert [r['row_begin']for r in q['ranks']]==[rank*5120 for rank in range(32)]
        registered.append(q);return 0
    def load(ctx,payload):
        assert ctx==0x12345678ABCDEF01
        q=json.loads(payload);assert len(q)==1
        for item in q:
            assert item['pointer']in live and item['shape']==[163840,7168]and item['dtype']==1
            loads.append(dict(item,CPU_owner_live_during_CAPI=True))
        copy[0]+=8;copy[1]+=8;copy[2]+=8*73400320
        return 0
    def stats(out,count):
        assert count==5
        for i,v in enumerate(copy):out[i]=v
        return 0
    lib=SimpleNamespace(modern_kimi_shard_head_expected_r9183=Fn(register),modern_kimi_shard_head_error_r9183=Fn(lambda:b'recorded'),
        modern_kimi_engine_load_r2475=Fn(load),modern_kimi_engine_finish_r2475=Fn(lambda ctx:0),modern_cpu_copy_stats_r2515=Fn(stats))
    class Shard:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def get_tensor(self,name):return W(name)
    with tempfile.TemporaryDirectory(prefix='r461-loader-recording-')as tmp:
        p=Path(tmp);keys={'language_model.model.embed_tokens.weight','language_model.lm_head.weight'}
        (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'recording-only.safetensors'for k in keys}}))
        result=load_parameters(lib,0x12345678ABCDEF01,p,{},keys,lambda *a,**k:Shard(),torch,
            lambda code:None if code==0 else (_ for _ in()).throw(RuntimeError(code)),lambda report:None)
    assert result['loaded_keys']==2 and result['loader_flushes']==2 and len(loads)==2 and len(registered)==1
    assert visited==[rank*5120+row for rank in range(32)for row in [0,2560,5119]]
    gc.collect();assert not live,'original Python pending owners released after sync CAPI returns'
    q=dict(passed=True,scope='actual new loader main/function with recording safe_open/Tensor/Native boundaries',
        actual_full_CPU_descriptors_preserved=True,high_context_pointer_preserved=True,registered_ranks=32,head_sample_rows=[0,2560,5119],
        actual_loaded_CPU_keys=sorted(keys),actual_flushes=2,checkpoint_weight_payload=False,Native_dlopen=False,SDK_execution=False,GPU=0,
        CPP_Weight_slice_and_H2D_ownership='Validated separately by actual byte-extracted Weight/LoadBatch source consumer, not these recording descriptor leaves')
    assert 'torch'not in sys.modules
    print(json.dumps(q,indent=2))
if __name__=='__main__':main()
