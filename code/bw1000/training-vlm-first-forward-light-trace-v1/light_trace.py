"""Opt-in hash/header observer. Disabled path does not query/import any backend."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

SCHEMA='VLM_FIRST_FORWARD_LIGHT_TRACE_V1'


def need(ok,reason):
    if not ok:raise ValueError(reason)


def digest(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def install_if_enabled(config,*,rank,backend_factory):
    # Deliberately before validation, path reads, module imports or model access.
    if config is None or config.get('enabled',False) is False:return None
    need(config.get('enabled') is True and config.get('schema')==SCHEMA,'explicit trace opt-in required')
    need(config.get('root_GO') is True and config.get('trace_GO') is True
         and config.get('diagnostic_only') is True,'trace permission/diagnostic scope required')
    need(config.get('ranks')==[0,1] and type(rank)is int and 0<=rank<8,'affected/control rank selection differs')
    if rank not in config['ranks']:return None
    need(config.get('phase') in ('continuous','resumed','independent_control'),'unknown trace phase')
    expected=81 if config['phase']=='continuous' else 1
    need(config.get('start_ordinal')==expected and config.get('count')==4,'fixed first accumulation selection required')
    need(type(config.get('request_id'))is str and len(config['request_id'])==32
         and set(config['request_id'])<=set('0123456789abcdef'),'new diagnostic request required')
    backend=backend_factory() # no invocation for disabled or unselected ranks
    return LightTrace(config,rank,backend)


class LightTrace:
    def __init__(self,config,rank,backend):
        self.config,self.rank,self.backend=dict(config),rank,backend
        self.ordinal=0;self.pending=None;self.entries=0;self.exits=0;self.failed=False
        self.handles=[];self.total_bytes=0;self.sequence=[];self.contexts={}
        self.root=Path(config['output'])/('rank-%02d'%rank)
        scope=Path(config['scope_root']).resolve(strict=True)
        need(self.root.is_absolute() and self.root.resolve().is_relative_to(scope)
             and not self.root.is_symlink(),'new private trace path outside scope')
        self.root.mkdir(mode=0o700,parents=False) # rank dir exclusive; caller prepares fresh output parent

    def save(self,name,value,cap=256*1024):
        raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
        need(cap in (256*1024,1024**2) and len(raw)<=cap and self.total_bytes+len(raw)<=4*1024**2,
             'private trace record/total cap')
        with (self.root/name).open('xb') as f:
            f.write(raw);f.flush();os.fsync(f.fileno())
        self.total_bytes+=len(raw)
        pin=dict(name=name,bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
        self.sequence.append(pin)
        return pin

    def capture(self,event,module,args,kwargs,output=None):
        self.backend.check()
        before=self.backend.rng_snapshot()
        digest(before) # only JSON-serializable hashes; raw RNG tensors cannot escape
        meta=self.backend.metadata(module,args,kwargs,output)
        after=self.backend.rng_snapshot()
        need(before==after,'metadata observation changed RNG; preserve failure')
        # Preserve the full raw context privately once per distinct value;
        # don't repeat thousands of module-mode names in each entry/exit.
        if 'context' in meta:
            context=meta['context'];key=digest(context)
            if key not in self.contexts:
                self.contexts[key]=self.save('context-'+key+'-private.json',context,cap=1024**2)
            meta=dict(meta,context=dict(identity_SHA256=key,private_record=self.contexts[key]))
        row=dict(schema=SCHEMA,request_id=self.config['request_id'],phase=self.config['phase'],rank=self.rank,
            ordinal=self.ordinal,tail_ordinal=self.ordinal-self.config['start_ordinal']+1,event=event,
            observed_monotonic=time.monotonic(),RNG=before,observer_RNG_unchanged=True,
            metadata=meta,model_tensor_payload_read=False,RNG_state_hashed_only=True,
            performance_qualified=False,recovery_qualified=False)
        self.save('%03d-%s-private.json'%(self.ordinal,event),row)
        return meta

    def before(self,module,args,kwargs):
        self.ordinal+=1
        start=self.config['start_ordinal']
        if not start<=self.ordinal<start+4:return None
        need(self.pending is None,'nested/repeated selected top-level forward')
        self.pending=self.ordinal
        self.entries+=1
        try:self.capture('entry',module,args,kwargs)
        except BaseException as e:
            self.fail(type(e).__name__,'entry');raise
        return None # never replace input/kwargs

    def after(self,module,args,kwargs,output):
        if self.pending is None:return None
        need(self.pending==self.ordinal,'selected forward exit does not match entry')
        try:
            meta=self.capture('exit',module,args,kwargs,output)
            self.exits+=1;self.pending=None
            if meta.get('native_exception_active') is not None:
                self.fail(meta['native_exception_active'],'native_exception')
        except BaseException as e:
            self.fail(type(e).__name__,'exit');raise
        return None # never replace native output, including None/exception path

    def attach(self,model):
        try:
            self.handles.append(model.register_forward_pre_hook(self.before,with_kwargs=True))
            self.handles.append(model.register_forward_hook(self.after,with_kwargs=True,always_call=True))
        except BaseException:
            self.detach();raise
        return self

    def detach(self):
        for handle in self.handles:handle.remove()
        self.handles=[]

    def fail(self,error_type,event):
        self.failed=True
        path=self.root/'failure-private.json'
        if not path.exists():
            self.save(path.name,dict(error_type=error_type,event=event,ordinal=self.ordinal,
                selected_entries_attempted=self.entries,selected_exits_completed=self.exits,
                planned_forward_pairs=4,request_id=self.config['request_id'],rank=self.rank))

    def finish(self):
        self.detach()
        complete=not self.failed and self.entries==self.exits==4 and self.pending is None
        row=dict(schema=SCHEMA,request_id=self.config['request_id'],phase=self.config['phase'],rank=self.rank,
            planned_forward_pairs=4,entries_attempted=self.entries,exits_completed=self.exits,
            complete=complete,records=self.sequence.copy(),total_private_bytes=self.total_bytes,
            native_choice_preserved=True,model_tensor_payload_read=False,raw_inputs_or_tokens_saved=False,
            raw_RNG_saved=False,GPU_work_started_by_observer=False,performance_qualified=False,recovery_qualified=False)
        self.save('result-private.json',row)
        return row
