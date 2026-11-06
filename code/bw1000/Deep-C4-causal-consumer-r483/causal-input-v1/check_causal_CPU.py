"""CPU recording of the caller, not Native/device/numeric evidence."""
import copy as CP
import ctypes as C
import json
import struct
import time
from causal import H,inputs,schedule,binding
from exercise import run

class Recording:
    def __init__(self,fault=None):
        self.fault=fault;self.past=0;self.pending=None;self.counts=dict(prepares=0,commits=0,aborts=0,resets=0)
        self.state=bytearray(8192);self.cache=bytearray(b'\xa5'*1081344);self.inputs=[]
    def read(self,name,*args):
        if name=='parameters':return [dict(name=str(i),loaded=self.fault!='unloaded') for i in range(7)]
        if name=='observe':return dict(past=self.past,pending=self.pending is not None,ready=True,
             local_heads=64,compressed_capacity=8192,full_context=32768,driver_free=64*1024**3,**self.counts)
        if name=='tensor':
            m,b=self.tensor(args[0].decode());return m
        raise AssertionError(name)
    def tensor(self,name):
        if name=='state':return dict(shape=[8,512],dtype=9,bytes=8192,contiguous=True),self.state
        if name=='raw_cache':return dict(shape=[128,8448],dtype=4,bytes=1081344,contiguous=True),self.cache
        assert self.pending is not None
        width=(self.past+self.pending)//4;n=min(512,width)
        if name=='scores':
            b=struct.pack('<'+'f'*width,*range(width))
            return dict(shape=[1,width],dtype=10,bytes=len(b),contiguous=True),b
        assert name=='indices';ids=list(range(width-1,width-n-1,-1))
        if self.fault=='indices':ids[0]=width
        b=struct.pack('<'+'q'*n,*ids)
        return dict(shape=[1,n],dtype=3,bytes=len(b),contiguous=True),b
    def call(self,name,*args):
        if name=='copy':
            _,b=self.tensor(args[0].decode());assert len(b)==args[3];C.memmove(args[2],bytes(b),len(b));return
        if name=='reset':
            assert self.pending is None;self.past=0;self.state[:]=b'\x00'*len(self.state);self.counts['resets']+=1
            if self.fault=='reset_cache':self.cache[0]^=1
            return
        if name=='prepare':
            past,n,h,hbytes,q,qbytes,pos,npos,seconds=args
            assert past==self.past and self.pending is None
            assert (hbytes,qbytes,npos)==(n*8192,n*2048,n)
            assert [pos[i] for i in range(n)]==list(range(past,past+n))
            self.inputs.append((past,n,H(h.raw),H(q.raw)));self.pending=n;self.counts['prepares']+=1
            C.cast(seconds,C.POINTER(C.c_double))[0]=0.000125
            if self.fault=='host_mutation':h[0]=b'\x00'
            return
        if name=='abort':
            assert self.pending is not None;self.pending=None;self.counts['aborts']+=1
            if self.fault=='abort_state':self.state[0]^=1
            if self.fault=='abort_cache':self.cache[0]^=1
            return
        if name=='commit':
            assert self.pending is not None;self.past+=self.pending;self.pending=None;self.counts['commits']+=1
            self.state[0]=(self.state[0]+1)%256;self.cache[0]=(self.cache[0]+1)%256
            C.cast(args[0],C.POINTER(C.c_double))[0]=0.0000625
            if self.fault=='past':self.past+=1
            return
        raise AssertionError(name)

def main():
    start=time.monotonic();p=binding();raw=json.dumps(p,sort_keys=True).encode()
    # Exercise JSON serialization/order independence, actual 69 input buffers and all fullcontext steps.
    loaded=json.loads(raw);t=Recording();saved=[];deadlines=[];events=[]
    result=run(t,loaded,deadlines.append,lambda label,name,m,b:saved.append((label,name,H(b))),events.append)
    assert result['counts']==dict(prepares=69,commits=68,aborts=1,resets=2) and len(t.inputs)==69
    assert len(saved)==6 and len(result['records'])==71 and len(deadlines)==140
    assert sum(s['tokens'] for s in schedule() if s['action']=='prepare')==33281
    assert sum(s['tokens'] for s in schedule() if s['action']=='prepare' and s['commit'])==33280
    assert inputs(0,512)[0]==inputs(0,17)[0]+inputs(17,495)[0]
    assert inputs(0,512)[0]!=inputs(0,512,1)[0]
    assert all((x&0x7f80)!=0x7f80 for x, in struct.iter_unpack('<H',inputs(0,1)[0]))
    rejected=[]
    for fault in ('unloaded','reset_cache','abort_state','abort_cache','indices','host_mutation','past'):
        try:run(Recording(fault),loaded,lambda x:None,lambda *x:None,lambda x:None)
        except ValueError:rejected.append(fault)
        else:raise AssertionError('accepted '+fault)
    for name,mutate in [('input_pin',lambda q:q['steps'][1]['inputs']['hidden_BF16'].update(SHA256='0'*64)),
                       ('context',lambda q:q.update(full_context=8192)),
                       ('floor',lambda q:q.update(floor_bytes=8*1024**3)),
                       ('topk',lambda q:q.update(topk=256)),
                       ('partial_schedule',lambda q:q['steps'].pop()),
                       ('noncausal',lambda q:q['steps'][2].update(past=1024))]:
        q=CP.deepcopy(loaded);mutate(q)
        try:run(Recording(),q,lambda x:None,lambda *x:None,lambda x:None)
        except ValueError:rejected.append(name)
        else:raise AssertionError('accepted '+name)
    failed_events=[]
    try:run(Recording('abort_state'),loaded,lambda x:None,lambda *x:None,failed_events.append)
    except ValueError:pass
    assert failed_events[0]['event']=='planned' and failed_events[-1]['event']=='failed'
    assert sum(x['event']=='step_complete' for x in failed_events)==18
    assert len(events)==73 and events[-1]['complete'] is True
    report=dict(passed=True,CPU_only=True,rejections=rejected,input_binding_SHA256=H((json.dumps(p,ensure_ascii=False,indent=2)+'\n').encode()),
        prepares=69,commits=68,aborts=1,resets=2,submitted_rows=33281,committed_rows=33280,
        sink_original_buffers_recorded=6,progress_events=73,failed_partial_steps_preserved=18,deadline_callbacks_recorded=len(deadlines),
        actual_Native_calls=0,actual_weights_read=0,actual_Runtime=0,actual_GPU=0,
        whole_CPU_wall_s=time.monotonic()-start)
    print(json.dumps(report,indent=2))
    return p,report

if __name__=='__main__':main()
