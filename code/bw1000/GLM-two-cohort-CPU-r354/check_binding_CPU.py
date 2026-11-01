"""Execute actual private forward binding against recording Native/Torch boundaries."""
import ctypes
import json
from types import SimpleNamespace
from native_binding import Forward,create_args


def check_binding():
    recordings=[]
    for stage in range(4):
        for rank in range(8):
            pointers={};calls=[]
            def tensor(values,**kw):
                return SimpleNamespace(values=tuple(values),kind='input',finite=True)
            def full(shape,value,**kw):
                return SimpleNamespace(shape=shape,kind='logits',finite=False)
            def ptr(t):
                pointers[id(t)]=t
                return ctypes.c_void_p(id(t))
            def native(ctx,tokens,count,desc,ids,pos,logits,all_positions,stream):
                row,past,n=tuple(desc)
                assert ctx==123 and (tokens,count,n,all_positions,stream)==(1,1,1,0,456)
                assert pointers[pos.value].values==(past,)
                assert pointers[ids.value].values==((101+row,) if stage==0 else (0,))
                assert pointers[logits.value].shape==(1,154880)
                pointers[logits.value].finite=True
                calls.append([row,past,n,tokens,count])
                return 0
            torch=SimpleNamespace(tensor=tensor,full=full,int64='I64',bfloat16='BF16',
                cuda=SimpleNamespace(synchronize=lambda r:None),
                isfinite=lambda t:SimpleNamespace(all=lambda:t.finite))
            f=Forward(SimpleNamespace(modern_glm_model_forward=native),123,stage,rank,torch,ptr,
                      lambda:456,lambda code:None if code==0 else (_ for _ in ()).throw(ValueError(code)))
            for row,past in [(0,8912),(1,32760)]:
                logits,*_=f(tokens=1,count=1,descriptor=(row,past,1),input_ids=(101+row,),positions=(past,))
                assert logits.finite
            cfg=json.dumps(dict(num_hidden_layers=78,hidden_size=6144,vocab_size=154880,max_position_embeddings=32768)).encode()
            args=create_args(rank,stage,cfg,'WORLD','TP','CONTROL')
            assert args[:4]==(rank,stage,2,32768) and args[5:]==('WORLD','TP','CONTROL')
            recordings.append(dict(stage=stage,rank=rank,create_rows=args[2],create_capacity=args[3],forward_metadata=calls))
    return {'passed':True,'actual_binding_module_executed':True,'Native_mock':True,'Torch_mock':True,
            'real_factory_called':False,'GPU':0,'owners':recordings}
