"""Symmetric RNG identity witness in both arms; full tracer is a separate opt-in."""
import json
import os
from pathlib import Path


class CommonRNGTap:
    def __init__(self,backend,root,rank):
        self.backend,self.rank=backend,rank
        self.root=Path(root)/('rank-%02d'%rank);self.root.mkdir(mode=0o700)
        self.ordinal=0;self.pending=False;self.rows=[];self.handles=[]

    def record(self,event):
        self.backend.check()
        before=self.backend.rng_snapshot()
        row=dict(ordinal=self.ordinal,event=event,rank=self.rank,
                 global_step=self.backend.trainer.state.global_step,RNG=before)
        wrapped=getattr(self.backend.trainer,'model_wrapped',None)
        def scalar(value):
            return value if value is None or type(value)in(bool,int,float,str) else {'type':type(value).__name__,'value_not_read':True}
        row['DDP_context']=dict(type=type(wrapped).__name__,flags={name:scalar(getattr(wrapped,name,None)) for name in
            ('require_backward_grad_sync','require_forward_param_sync','broadcast_buffers','find_unused_parameters',
             'gradient_as_bucket_view','static_graph','_has_rebuilt_buckets')})
        if row['global_step']!=20:raise ValueError('common tap outside first accumulation at20')
        raw=(json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode()
        if len(raw)>16384:raise ValueError('common RNG metadata bound')
        with (self.root/('%02d-%s.json'%(self.ordinal,event))).open('xb') as f:
            f.write(raw);f.flush();os.fsync(f.fileno())
        if before!=self.backend.rng_snapshot():raise ValueError('common tap changed RNG')
        self.rows.append(row)

    def before(self,module,args,kwargs):
        self.ordinal+=1
        if self.ordinal<=4:
            if self.pending:raise ValueError('nested common tap forward')
            self.pending=True;self.record('entry')
        return None

    def after(self,module,args,kwargs,output):
        if self.pending:
            self.record('exit');self.pending=False
        return None

    def attach(self,model):
        try:
            self.handles.append(model.register_forward_pre_hook(self.before,with_kwargs=True))
            self.handles.append(model.register_forward_hook(self.after,with_kwargs=True,always_call=True))
        except BaseException:
            for h in self.handles:h.remove()
            raise
        return self

    def finish(self):
        for h in self.handles:h.remove()
        complete=len(self.rows)==8 and not self.pending
        result=dict(rank=self.rank,complete=complete,planned_forward_pairs=4,rows=self.rows,
                    model_tensor_payload_read=False,raw_RNG_saved=False)
        with (self.root/'result.json').open('x') as f:json.dump(result,f,sort_keys=True,allow_nan=False)
        if not complete:raise ValueError('common RNG first4 denominator incomplete')
        return result
