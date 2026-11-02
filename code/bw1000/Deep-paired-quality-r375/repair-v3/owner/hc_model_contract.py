"""Default-OFF real Native CAPI mode and per-device HC coverage, no GPU readback."""
import ctypes,time
FIELDS=('created','ready','pre','eligible_bf16_t1','completed','other_t','f32_t1')
class HCModel:
    def __init__(self,lib,layers):
        assert layers==43;self.expected=layers*2;self.lib=lib
        self.set=lib.deep_hc_model_epilogue_set_r366;self.set.argtypes=[ctypes.c_int];self.set.restype=ctypes.c_int
        self.get=lib.deep_hc_model_epilogue_get_r366;self.get.argtypes=[];self.get.restype=ctypes.c_int
        self.read=lib.deep_hc_model_epilogue_counters_r366;self.read.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_int];self.read.restype=ctypes.c_int
        assert self.get()==0,'New Native HC mode must be defaultOFF before first setter'
        self.initial=self.snapshot();assert all(row[:2]==[86,86]for row in self.initial),'Actual Model must have86constructed/loadedHCinstances on each owner'
    def snapshot(self):
        out=(ctypes.c_uint64*56)();assert self.read(out,56)==0
        return [[int(out[d*7+k])for k in range(7)]for d in range(8)]
    def select(self,mode):
        assert type(mode)is int and mode in [0,1]and self.set(mode)==0 and self.get()==mode;return self.snapshot()
    def verify_forward(self,before,after,tokens,mode):
        assert type(tokens)is int and tokens>=1 and mode in [0,1]
        expected=[0,0,86,86 if tokens==1 else 0,86 if tokens==1 and mode else 0,86 if tokens!=1 else 0,0]
        delta=[]
        for old,new in zip(before,after):
            assert old[:2]==new[:2]==[86,86];d=[b-a for a,b in zip(old,new)];assert d==expected,('HC366 actual forward coverage differs',tokens,mode,d,expected);delta.append(d)
        return dict(input_tokens=tokens,mode=mode,per_owner_delta=delta,counter_fields=FIELDS,eligible_T1_BF16=tokens==1,candidate_completed_branch=tokens==1 and mode==1,not_hardware_GPU_hit_counter=True)
