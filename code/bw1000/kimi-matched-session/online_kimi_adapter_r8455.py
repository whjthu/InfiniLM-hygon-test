"""Actual existing CAPI adapter for an ALREADY loaded private Kimi Engine.

NOT EXECUTED ON GPU. Loading, source/DSO/SDK closure, per-owner visible8GPU init,
weight finalization/QA, memory budget, external watchdog and postKFD are runner
responsibilities. This skeleton never dlopens a library or invents a context.
"""
import ctypes
import threading


class KimiLoadedAdapter:
    def __init__(self,lib,ctx,*,stage,stages,rows,check,memory_gate,destroy):
        assert threading.current_thread() is threading.main_thread()
        assert ctx and 0<=stage<stages==4 and 1<=rows<=32
        assert callable(check) and callable(memory_gate) and callable(destroy)
        import numpy as np
        self.np,self.lib,self.ctx=np,lib,ctx
        self.stage,self.stages,self.rows=stage,stages,rows
        self.check,self.memory_gate,self.destroy=check,memory_gate,destroy
        self.next_seq,self.poisoned,self.closed=0,False,False
        self.pasts=[0]*(rows+1)
        self.fn=lib.modern_kimi_engine_forward_r2475
        self.fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_int]+[ctypes.c_void_p]*9
        self.fn.restype=ctypes.c_int

    def forward(self,job):
        assert threading.current_thread() is threading.main_thread()
        assert not self.poisoned and not self.closed and job['seq']==self.next_seq
        np=self.np
        rows,pasts,inputs=job['rows'],job['pasts'],job['inputs']
        assert len(rows)==len(pasts)==len(inputs) and 0<len(rows)<=self.rows
        assert len(set(rows))==len(rows) and all(type(r)is int and 1<=r<=self.rows for r in rows)
        assert all(p==self.pasts[r] for r,p in zip(rows,pasts))
        widths=[len(v) for v in inputs]
        assert 0<sum(widths)<=8192 and all(w>0 for w in widths)
        if job['kind']=='prefill':assert all(p==0 for p in pasts)
        else:assert job['kind']=='decode' and all(w==1 and p>0 for w,p in zip(widths,pasts))
        assert all(p+w<=1024 for p,w in zip(pasts,widths))
        assert all(type(v)is int and 0<=v<163840 for seq in inputs for v in seq)
        arrays=[np.asarray(rows,np.int32),np.asarray(pasts,np.int32),
                np.asarray([p+w for p,w in zip(pasts,widths)],np.int32),
                np.asarray([0]+np.cumsum(widths).tolist(),np.int32),
                np.ascontiguousarray(np.concatenate(inputs),dtype=np.int64),
                np.ascontiguousarray(np.concatenate([np.arange(p,p+w,dtype=np.int64) for p,w in zip(pasts,widths)]))]
        output=np.full(len(rows),-999,np.int64);flags=np.asarray([-1],np.int32)
        self.memory_gate()
        # Own every CPU buffer until synchronous local return. NO global control
        # barrier here: PP jobs progress with identical immutable local order.
        try:
            self.check(self.fn(self.ctx,sum(widths),len(rows),71,
                              *[ctypes.c_void_p(v.ctypes.data) for v in arrays],
                              None,ctypes.c_void_p(output.ctypes.data),ctypes.c_void_p(flags.ctypes.data)))
            assert flags[0]==(2 if self.stage==self.stages-1 else 0)
            self.memory_gate()
            for r,p,w in zip(rows,pasts,widths):self.pasts[r]=p+w
            self.next_seq+=1
            if self.stage==self.stages-1:
                assert all(0<=v<163840 for v in output)
                return output.tolist()
            return None
        except BaseException:
            self.poisoned=True # never retry unknown partial native execution.
            raise

    def close(self):
        assert threading.current_thread() is threading.main_thread()
        if not self.closed:
            self.destroy(self.ctx) # supplied actual CAPI checked destroy/drain.
            self.closed=True
