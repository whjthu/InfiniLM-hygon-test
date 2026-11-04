"""Explicit future-owner argument contract; no loading, Native or GPU calls.

This is not a public service or a device-qualified profile. Existing TP16PP2
public entry remains unchanged. Only the new Native opt-in CAPI consumes it.
"""
from dataclasses import dataclass
import ctypes as C
@dataclass(frozen=True)
class Profile:
    name:str
    tp:int
    pp:int
    @property
    def tp_nodes(self):return self.tp//8
    @property
    def emitter_node(self):return (self.pp-1)*self.tp_nodes
    def rank(self,node,device):
        if not(0<=node<4 and 0<=device<8):raise ValueError('four real nodes/eight owners required')
        stage=node//self.tp_nodes;rank=(node%self.tp_nodes)*8+device
        cuts={4:(0,24,47,71,93),1:(0,93)}[self.pp]
        return dict(stage=stage,tp_rank=rank,world_rank=stage*self.tp+rank,layer_begin=cuts[stage],layer_end=cuts[stage+1],head_row_begin=rank*(163840//self.tp),head_row_end=(rank+1)*(163840//self.tp),local_MLA_heads=96//self.tp,local_KDA_heads=96//self.tp,local_expert_I=3072//self.tp,local_shared_I=6144//self.tp,tail=stage+1==self.pp,emitter_node=node==self.emitter_node)
    def create_arguments(self,raw,node,address,port,env):
        if env.get('INFINILM_KIMI_LONG_GEOMETRY_R453')!='1':raise ValueError('explicit defaultOFF opt-in required')
        self.rank(node,0)
        if not isinstance(raw,bytes) or not raw or not isinstance(address,bytes) or not address or not(1<=port<=65535):raise ValueError('typed CAPI arguments')
        return(raw,node,self.tp_nodes,self.pp,address,port)
PROFILES={'kimi-long-tp8pp4-ar-r453':Profile('kimi-long-tp8pp4-ar-r453',8,4),'kimi-long-tp32pp1-ar-r453':Profile('kimi-long-tp32pp1-ar-r453',32,1)}
CREATE_ARGTYPES=(C.c_char_p,C.c_int,C.c_int,C.c_int,C.c_char_p,C.c_int)
FORWARD_ARGTYPES=(C.c_void_p,C.c_int,C.c_size_t,C.c_size_t,C.c_int,*([C.POINTER(C.c_int32)]*4),*([C.POINTER(C.c_int64)]*2),C.c_void_p,C.c_void_p,C.POINTER(C.c_int))
LIMITS=dict(logical=1,physical=2,state_pool=3,context=32768,prompt_chunk=512,output_budget=4096,precreate_allowner_floor_bytes=4*1024**3,AR_only=True,capture=False,replay=False,GPU_GO=False,Model_GO=False)
def select(name,*,optin=False):
    if not optin or name not in PROFILES:raise ValueError('explicit named r453 profile only')
    return PROFILES[name]
