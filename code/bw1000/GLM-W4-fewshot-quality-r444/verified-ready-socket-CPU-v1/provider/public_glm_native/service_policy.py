"""Explicit private NoV16 profiles; no universal default or Native imports."""
from pathlib import Path
import contextvars,hashlib,json,os
DEFAULT_PROFILE='glm-nov16-w4'
WINDOWS={'glm-nov16-w2':2,'glm-nov16-w4':4}
NATIVE_SHA256='9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'
PARENT_OWNER_SHA256='5e944fb676f15022f1387d936cc9286c00953f0ca6884ef79b12979b4bca8ac4'
PROFILE_MAGIC=629417
_REQUEST_PROFILE=contextvars.ContextVar('glm_r417_request_profile',default=None)
def select(data,default=DEFAULT_PROFILE):
    profile=data.get('native_profile',default)
    if type(profile)is not str or profile not in WINDOWS:raise ValueError('native_profile must be glm-nov16-w2 or glm-nov16-w4')
    return profile,WINDOWS[profile]
def request_profile(data,default):
    if 'native_profile'in data:raise ValueError('Select the private profile with X-InfiniLM-Native-Profile header;request body is not rewritten')
    return select({'native_profile':_REQUEST_PROFILE.get()or default})
class ProfileHeaderMiddleware:
    """One app instance, ASGI header context only; body and route untouched."""
    def __init__(self,app,default=DEFAULT_PROFILE):self.app=app;self.default=default;select({},default)
    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        try:
            values=[v for k,v in scope.get('headers',[])if k.lower()==b'x-infinilm-native-profile']
            if len(values)>1:raise ValueError('Duplicate Native profile header')
            profile=values[0].decode('ascii')if values else self.default;select({'native_profile':profile})
        except(ValueError,UnicodeError)as error:
            body=json.dumps({'error':str(error)}).encode();await send({'type':'http.response.start','status':400,'headers':[(b'content-type',b'application/json')]});await send({'type':'http.response.body','body':body});return
        token=_REQUEST_PROFILE.set(profile)
        try:return await self.app(scope,receive,send)
        finally:_REQUEST_PROFILE.reset(token)
def bind_prequeue(engine,default=DEFAULT_PROFILE):
    """One prepared engine instance; reject invalid policy before queue/Native."""
    select({},default);original=engine.prequeue_factory;validate=engine.validate_data
    if original is None:raise ValueError('Full prepared history prequeue is required')
    if getattr(engine,'_r417_bound',False):raise ValueError('Profile already bound')
    def check(data,params):
        request_profile(data,default)
        if type(params.max_tokens)is not int or not 1<=params.max_tokens<=4096:raise ValueError('Explicit NoV16 profile budget must be integer1..4096;no clamp')
        return validate(data,params)
    def prequeue(req,messages,data,generation,kwargs):
        profile,window=request_profile(data,default)
        if req.max_new_tokens>4096:raise ValueError('Explicit NoV16 profile ordinary budget is at most4096;no clamp')
        bound=original(req,messages,data,generation,kwargs)
        bound.update(native_profile=profile,prefill_control_window=window,profile_policy='r417-explicit-NoV16',math_or_template_changed=False)
        return bound
    engine.validate_data=check;engine.prequeue_factory=prequeue;engine._r417_bound=True
def encode_profile_wire(seq,op,ids=(),limit=0,profile=None):
    from .longcontext_wire_r204 import encode_wire,BEGIN
    values=encode_wire(seq,op,ids,limit)
    if op==BEGIN:
        _,window=select({'native_profile':profile});assert values[-2:]==[0,0]
        values[-2:]=[PROFILE_MAGIC,window]
    elif profile is not None:raise ValueError('Control wire cannot select a request profile')
    return values
def decode_profile_wire(values,seq,vocab):
    from .longcontext_wire_r204 import decode_wire,BEGIN
    values=list(values);window=None
    if len(values)>=5 and values[2]==BEGIN:
        if values[-2]!=PROFILE_MAGIC or type(values[-1])not in(int,float)or values[-1]not in(2,4):raise ValueError('Missing or invalid explicit profile wire')
        window=int(values[-1]);values[-2:]=[0,0]
    op,prompt,limit=decode_wire(values,seq,vocab)
    return op,prompt,limit,window
def verify_sources(directory,parent):
    directory=Path(directory);parent=Path(parent);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    m=json.loads((directory/'SOURCE_MANIFEST.json').read_bytes())
    for n,h in m['files_SHA256'].items():
        p=directory/n
        if not p.resolve().is_relative_to(directory.resolve())or sha(p)!=h:raise ValueError('Private source mismatch:'+n)
    if sha(parent/'SOURCE_MANIFEST.json')!=PARENT_OWNER_SHA256:raise ValueError('Qualified loader parent mismatch')
    p=json.loads((parent/'SOURCE_MANIFEST.json').read_bytes())
    for n,h in p['files_SHA256'].items():
        if sha(parent/n)!=h:raise ValueError('Qualified parent member mismatch:'+n)
    return m
def configure(profile,environ):
    """Call only from this explicit private entry; absence leaves old entry alone."""
    select({'native_profile':profile})
    fixed={'GLM_PP_SIZE_R3603':'2','GLM_CACHE_ROWS_R3627':'1','GLM_CACHE_CAP_R3627':'32768','GLM_V_ON_DEMAND_R403':'1','GLM_BOUNDED_ROPE_R4279':'1','GLM_SHARED_CONTROL_R4279':'1','GLM_R417_DEFAULT_PROFILE':profile}
    for k,v in fixed.items():
        if k in environ and environ[k]!=v:raise ValueError('Conflicting explicit profile environment:'+k)
    environ.update(fixed)
    return fixed
def stop_requested(path,run,profile):
    path=Path(path)
    if not path.exists():return False
    q=json.loads(path.read_bytes())
    if q!={'normal_stop':True,'run':run,'profile':profile}:raise ValueError('Normal stop marker identity mismatch')
    return True
