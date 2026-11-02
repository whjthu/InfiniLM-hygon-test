"""Ctypes declarations copied from existing public constructor consumers only."""
import base64,ctypes as C,json,os
from pathlib import Path

SYMBOLS=['modern_glm_model_error','modern_glm_model_id_size','modern_glm_model_id','modern_glm_model_create',
         'modern_glm_model_destroy','modern_glm_model_shutdown','modern_glm_partition_report_r3854',
         'modern_glm_constructor_census_r4158','modern_glm_unloaded_trim_r4052','modern_glm_model_control']

def declare(lib):
    P=C.c_void_p;S=C.c_size_t
    lib.modern_glm_model_error.restype=C.c_char_p
    lib.modern_glm_model_id_size.argtypes=[];lib.modern_glm_model_id_size.restype=S
    lib.modern_glm_model_id.argtypes=[P]
    lib.modern_glm_model_create.argtypes=[C.c_int,C.c_int,S,S,C.c_char_p,P,P,P];lib.modern_glm_model_create.restype=P
    lib.modern_glm_model_destroy.argtypes=[P];lib.modern_glm_model_shutdown.argtypes=[]
    for name in ['modern_glm_partition_report_r3854','modern_glm_constructor_census_r4158']:
        f=getattr(lib,name);f.argtypes=[P];f.restype=C.c_char_p
    lib.modern_glm_unloaded_trim_r4052.argtypes=[P,C.POINTER(C.c_uint64),S]
    lib.modern_glm_model_control.argtypes=[P,S,P,P]
    return lib

def unique(lib):
    size=lib.modern_glm_model_id_size();assert 0<size<4096
    out=C.create_string_buffer(size);check(lib,lib.modern_glm_model_id(out))
    return base64.b64encode(out.raw).decode()

def check(lib,code):
    if code:raise RuntimeError(lib.modern_glm_model_error().decode())

def providers(lib,expected):
    class Info(C.Structure):_fields_=[('fname',C.c_char_p),('base',C.c_void_p),('sname',C.c_char_p),('addr',C.c_void_p)]
    dl=C.CDLL(None).dladdr;dl.argtypes=[C.c_void_p,C.POINTER(Info)];dl.restype=C.c_int
    result={}
    for name in SYMBOLS:
        info=Info();assert dl(C.cast(getattr(lib,name),C.c_void_p),C.byref(info))
        actual=Path(info.fname.decode()).resolve();assert actual==Path(expected).resolve(),name
        result[name]=str(actual)
    return result

class Device:
    def __init__(self,lib,torch,rank,affinity_reader=None):self.lib,self.torch,self.rank=lib,torch,rank;self.affinity_reader=affinity_reader or (lambda rank:sorted(os.sched_getaffinity(0)))
    def create(self,rank,stage,rows,capacity,config,payload):
        buffers=[C.create_string_buffer(base64.b64decode(payload[k] if k!='tp' else payload[k][stage])) for k in ['world','tp','control']]
        return self.lib.modern_glm_model_create(rank,stage,rows,capacity,json.dumps(config).encode(),*buffers)
    def partition(self,ctx):return json.loads(self.lib.modern_glm_partition_report_r3854(ctx))
    def census(self,ctx):return json.loads(self.lib.modern_glm_constructor_census_r4158(ctx))
    def trim(self,ctx):
        a=(C.c_uint64*6)();check(self.lib,self.lib.modern_glm_unloaded_trim_r4052(ctx,a,6));return list(a)
    def identity_sum(self,ctx):
        t=self.torch.ones(1,device='cuda',dtype=self.torch.float32)
        check(self.lib,self.lib.modern_glm_model_control(ctx,1,C.c_void_p(t.data_ptr()),C.c_void_p(self.torch.cuda.current_stream().cuda_stream)))
        value=t.item();del t;return value
    def affinity(self,rank):
        masks=self.affinity_reader(rank);assert len(set(masks))==4
        return dict(rank=rank,CPUs=masks,width=4,scope='Actual OS owner-process affinity readback after original Runtime hook;CPUtests record thisleaf')
    def snapshot(self):
        t=self.torch;t.cuda.synchronize(self.rank);free,total=t.cuda.mem_get_info(self.rank)
        return dict(free_bytes=free,total_bytes=total,driver_used_bytes=total-free,
                    Torch_allocated_bytes=t.cuda.memory_allocated(self.rank),Torch_reserved_bytes=t.cuda.memory_reserved(self.rank),
                    Native_allocator_allocated_bytes=None,Native_allocator_reserved_bytes=None,is_peak=False)
    def destroy(self,ctx):check(self.lib,self.lib.modern_glm_model_destroy(ctx))
    def shutdown(self):check(self.lib,self.lib.modern_glm_model_shutdown())
