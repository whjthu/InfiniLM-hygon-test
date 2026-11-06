"""Explicit typed TP16PP1 caller. Importing this module never loads a DSO."""
import ctypes as C
import json
import threading

V=C.c_void_p
Z=C.c_size_t
I=C.c_int
P=C.c_char_p

SIGNATURES={
 'id_size':([],Z), 'id':([V,Z],I),
 'create':([I,I,Z,Z,P,V,V,Z],V), 'error':([],P),
 'keys':([V],P), 'load_batch':([V,P],I), 'finish':([V],I),
 'forward':([V,Z,Z,C.POINTER(Z),V,V,V,I,V],I),
 'control':([V,Z,V,V],I), 'reset':([V],I),
 'geometry':([V],P), 'destroy':([V],I),
}

def attach(library):
    result={}
    for name,(args,ret) in SIGNATURES.items():
        f=getattr(library,'modern_glm_global_'+name+'_r470')
        f.argtypes=args;f.restype=ret;result[name]=f
    return result

class GlobalTP16Context:
    """Same-thread strong owner. CPU sources survive completion-returning load.

    `library` must already have been mapped and independently SHA/dladdr checked
    by the future deployment. This API does not inherit old952 Model readiness.
    No pre-sharding: original Weight::select_source uses global tp_rank/size.
    """
    def __init__(self,library,config,node,device,world_uid,tp_uid):
        if type(node)is not int or node not in (0,1) or type(device)is not int or not 0<=device<8:
            raise ValueError('two nodes / local CUDA device0..7 required')
        if not config.get('glm_global_tp16_r470') or not config.get('glm_global_tp16_Model_r470'):
            raise ValueError('both explicit private JSON optins required')
        self.library=library;self.f=attach(library);self.thread=threading.get_ident()
        self.rank=node*8+device;self.past=0;self.ready=False;self.poisoned=False;self.ctx=None
        n=int(self.f['id_size']())
        if not isinstance(world_uid,bytes)or not isinstance(tp_uid,bytes)or n<=0 or len(world_uid)!=n or len(tp_uid)!=n or world_uid==tp_uid:
            raise ValueError('two distinct exact-sized UIDs required')
        # The buffers and config survive the synchronous constructor; pointer is
        # returned as void*, never int, including above the signed32bit range.
        self._world=C.create_string_buffer(world_uid);self._tp=C.create_string_buffer(tp_uid)
        self.config_bytes=json.dumps(config,separators=(',',':')).encode()
        self.ctx=self.f['create'](node,device,1,32768,self.config_bytes,self._world,self._tp,n)
        if not self.ctx:raise RuntimeError(self.error())
    def owned(self,allow_poison=False):
        if threading.get_ident()!=self.thread:raise RuntimeError('owner thread differs')
        if not self.ctx or self.poisoned and not allow_poison:raise RuntimeError('closed/poisoned context')
    def error(self):
        b=self.f['error']();return b.decode(errors='replace')if b else 'Native error without text'
    def check(self,rc):
        if rc:self.poisoned=True;raise RuntimeError(self.error())
    def geometry(self):
        self.owned(True);b=self.f['geometry'](self.ctx)
        if not b:raise RuntimeError(self.error())
        q=json.loads(b)
        for k,v in {'TP':16,'PP':1,'world_size':16,'world_rank':self.rank,'tp_rank':self.rank,'rows':1,'capacity_per_row':32768,'local_attention_heads':4,'local_index_heads':2,'root_output_rank':0}.items():
            if q[k]!=v:raise ValueError('actual geometry '+k)
        return q
    def keys(self):
        self.owned();b=self.f['keys'](self.ctx)
        if not b:raise RuntimeError(self.error())
        keys=json.loads(b)
        if not isinstance(keys,list)or len(keys)!=len(set(keys))or any(not isinstance(k,str)for k in keys):raise ValueError('actual checkpoint keys')
        return keys
    def load_batch(self,records):
        self.owned()
        if self.ready:raise ValueError('already finalized')
        records=tuple(records);names=set();desc=[];held=[]
        for r in records:
            # An existing CPU checkpoint loader supplies a strong tensor/mmap
            # owner alongside its full, unsharded source shape/pointer.
            name=r['name'];shape=r['shape'];dtype=r['dtype'];pointer=r['pointer'];owner=r['owner']
            if not isinstance(name,str)or name in names or type(dtype)is not int or dtype not in (0,1,2)or type(pointer)is not int or pointer<=0 or not isinstance(shape,(list,tuple))or not shape or any(type(n)is not int or n<=0 for n in shape)or owner is None:
                raise ValueError('full source descriptor / strong owner')
            names.add(name);held.append(owner);desc.append(dict(name=name,dtype=dtype,shape=list(shape),pointer=pointer))
        if not desc:raise ValueError('empty batch')
        payload=json.dumps(desc,separators=(',',':')).encode()
        # Original load_batch performs copy_from+sync before returning. Neither
        # buffers nor the source owner can be released before check returns.
        self.check(self.f['load_batch'](self.ctx,payload))
        assert len(held)==len(desc)
    def finish(self):
        self.owned();self.check(self.f['finish'](self.ctx));self.ready=True
    def forward(self,tokens,past,input_pointer,position_pointer,logits_pointer,stream_pointer):
        self.owned()
        if not self.ready or type(tokens)is not int or not 1<=tokens<=4096 or type(past)is not int or past!=self.past or tokens>32768-past or any(type(p)is not int or p<=0 for p in (input_pointer,position_pointer,logits_pointer))or type(stream_pointer)is not int or stream_pointer<0:
            raise ValueError('C1 row0 / exact past / full32K / pointers')
        packet=(Z*3)(0,past,tokens)
        self.check(self.f['forward'](self.ctx,tokens,1,packet,V(input_pointer),V(position_pointer),V(logits_pointer),0,V(stream_pointer)))
        self.past+=tokens
    def control(self,count,pointer,stream_pointer):
        self.owned()
        if type(count)is not int or count<=0 or type(pointer)is not int or pointer<=0:raise ValueError('control buffer')
        self.check(self.f['control'](self.ctx,count,V(pointer),V(stream_pointer)))
    def reset(self):
        self.owned();self.check(self.f['reset'](self.ctx));self.past=0
    def close(self):
        self.owned(True);self.check(self.f['destroy'](self.ctx));self.ctx=None;self.ready=False
