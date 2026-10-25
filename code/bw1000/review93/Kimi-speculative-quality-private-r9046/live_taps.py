"""Host-owned feature copies from the measured native engine; no raw file IO."""
import ctypes as C
class LiveTaps:
    def __init__(self,lib,ctx,node):
        self.lib,self.ctx,self.node=lib,ctx,node;self.epoch=0
        lib.modern_kimi_live_taps_begin_r9046.argtypes=[C.c_void_p,C.c_uint64,C.c_int]
        lib.modern_kimi_live_taps_count_r9046.argtypes=[C.c_void_p,C.POINTER(C.c_size_t)]
        lib.modern_kimi_live_taps_info_r9046.argtypes=[C.c_void_p,C.c_size_t,C.POINTER(C.c_int64),C.c_size_t]
        lib.modern_kimi_live_taps_copy_r9046.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t]
    def begin(self,on):
        self.epoch+=1
        if self.lib.modern_kimi_live_taps_begin_r9046(self.ctx,self.epoch,int(on)):raise RuntimeError('live begin')
    def read(self,past,rows):
        import numpy as np
        n=C.c_size_t();assert self.lib.modern_kimi_live_taps_count_r9046(self.ctx,C.byref(n))==0
        expected={0:[19,37],2:[66,78,90]}.get(self.node,[]);assert n.value==len(expected),(self.node,n.value)
        records=[];data=[]
        for i in range(n.value):
            z=(C.c_int64*6)();assert self.lib.modern_kimi_live_taps_info_r9046(self.ctx,i,z,6)==0
            epoch,layer,begin,tokens,hidden,bytes_=list(z);assert epoch==self.epoch and begin==past and tokens==rows and hidden==7168 and bytes_==rows*7168*2
            raw=(C.c_uint8*bytes_)();assert self.lib.modern_kimi_live_taps_copy_r9046(self.ctx,i,raw,bytes_)==0;b=bytes(raw);v=np.frombuffer(b,dtype='<u2');assert not np.any((v&0x7f80)==0x7f80)
            records.append({'layer':layer,'begin':begin,'rows':tokens,'bytes':bytes_});data.append(b)
        assert sorted(z['layer']for z in records)==expected
        return {'records':records,'node':self.node,'epoch':self.epoch},b''.join(data)
def assemble(packets,past,rows):
    import numpy as np
    values={}
    for meta,raw in packets:
        pos=0
        for z in meta['records']:
            assert z['begin']==past and z['rows']==rows and z['bytes']==rows*7168*2 and z['layer']not in values
            values[z['layer']]=np.frombuffer(raw[pos:pos+z['bytes']],dtype='<u2').reshape(rows,7168);pos+=z['bytes']
        assert pos==len(raw)
    assert sorted(values)==[19,37,66,78,90]
    return np.concatenate([values[k]for k in [19,37,66,78,90]],axis=-1).tobytes()
