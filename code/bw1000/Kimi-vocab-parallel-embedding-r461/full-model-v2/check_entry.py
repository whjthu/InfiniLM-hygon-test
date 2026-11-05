"""Changed Python typed ABI/glue only; real libffi, recording Native/peer leaves."""
from pathlib import Path
from types import SimpleNamespace
import ast,ctypes as C,json,time
import numpy as np
import model_driver as d

def one(fail_at=None,expired=False):
    ctx=0x12345678abcdef01;calls=[];errors=[];keep=[];lib=SimpleNamespace()
    def invoke(pointer,kind,tokens,count,flags,*args):
        try:
            assert pointer==ctx and count==1 and flags==3
            assert C.cast(args[0],C.POINTER(C.c_int32))[0]==1
            past=C.cast(args[1],C.POINTER(C.c_int32))[0]
            assert C.cast(args[2],C.POINTER(C.c_int32))[0]==past+tokens
            off=C.cast(args[3],C.POINTER(C.c_int32));assert[off[0],off[1]]==[0,tokens]
            ids=C.cast(args[4],C.POINTER(C.c_int64));pos=C.cast(args[5],C.POINTER(C.c_int64))
            assert all(pos[i]==past+i and 0<=ids[i]<163840 for i in range(tokens))
            calls.append((kind,past,tokens))
            if len(calls)==fail_at:return 1
            words=(C.c_uint16*163840).from_address(args[6]);np.ctypeslib.as_array(words).fill(0x3f80)
            C.cast(args[7],C.POINTER(C.c_int64))[0]=163585 if kind==1 else 17
            C.cast(args[8],C.POINTER(C.c_int32))[0]=3
            return 0
        except BaseException as error:errors.append(repr(error));return 1
    f=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_int,C.c_size_t,C.c_size_t,C.c_int,*([C.c_void_p]*9))(invoke)
    setattr(lib,d.SYMBOL,f);keep.append(f);resets=[]
    def reset(pointer):
        assert pointer==ctx;resets.append(pointer);return 0
    for name in ['reset','finish']:
        f=C.CFUNCTYPE(C.c_int,C.c_void_p)(reset);setattr(lib,'modern_kimi_engine_'+name+'_r2475',f);keep.append(f)
    for name in ['modern_kimi_engine_load_r2475','modern_kimi_shard_head_expected_r9183','modern_kimi_shard_head_error_r9183']:
        setattr(lib,name,SimpleNamespace())
    lib.modern_kimi_engine_error_r2475=lambda:b'recorded failure'
    class Control:
        def collect(self,label,p):
            if label.startswith('typed-forward:'):return[dict(p,node=i)for i in range(4)]
            return[dict(p,node=i)for i in range(4)]
    failed=None;result=None
    try:result=d.generate(lib,ctx,0,Control(),[i%163840 for i in range(1025)],4096,time.monotonic()+(-1 if expired else 30),lambda *x:None)
    except(RuntimeError,AssertionError)as error:failed=repr(error)
    assert not errors,errors
    assert len(resets)==2
    if fail_at:assert failed and len(calls)==fail_at
    elif expired:assert failed and not calls
    else:
        assert calls==[(0,0,512),(0,512,512),(0,1024,1),(1,1025,1)]
        assert result['generated_ids']==[17,163585]and result['natural_EOS']and result['normal_reset']
    return dict(passed=True,forward_intervals=calls,reset_calls=len(resets),expected_failure=failed,high64_context_preserved=True)

def main():
    root=Path(__file__).resolve().parent
    for p in root.glob('*.py'):ast.parse(p.read_bytes())
    q=json.loads((root/'config.json').read_bytes());assert len(set(q['order']))==256 and q['order'][:2]==[0,128]
    node=(root/'node.py').read_text();assert node.index('if args.metadata_only:')<node.index('import torch')
    assert 'load_parameters(lib,ctx' in node and 'model_driver.generate(lib,ctx' in node
    assert 'embedding.run('not in node and 'modern_kimi_batch_'not in(root/'model_driver.py').read_text()
    return dict(passed=True,GPU=0,Native_dlopen=False,SDK_execution=False,real_libffi=True,recording_leaves=['Native functions','four-peer control','synthetic IDs for interval boundary only; production uses unchanged real r448 IDs'],cases=[one(),one(fail_at=2),one(expired=True)],old_contracts_reexecuted=False)
if __name__=='__main__':print(json.dumps(main(),indent=2))
