"""Binding only: existing current b4 C API. Does not load a DSO or create Runtime."""
import ctypes as C,json

def selected_ordinals_for_trace8(C_requests,prefill_micro=2):
    if type(C_requests)is not int or not 1<=C_requests<=16 or prefill_micro!=2:raise ValueError('qualified cap16/micro2')
    n=(C_requests+1)//2
    # Two warm verify RUNs; sample trace cycle 1,4,8. State-only controls are not RUN.
    return [n+2,n+5,n+9]

class ModelEvents:
    def __init__(self,already_loaded_lib,prepared_at_runtime_create=False):
        if not prepared_at_runtime_create:raise ValueError('ENV optin must precede original Runtime creation')
        self.configure_fn=already_loaded_lib.modern_kimi_model_events_configure_r8669zq
        self.configure_fn.argtypes=[C.c_char_p];self.configure_fn.restype=C.c_char_p
        self.read_fn=already_loaded_lib.modern_kimi_model_events_read_r8669zq
        self.read_fn.argtypes=[];self.read_fn.restype=C.c_char_p
        self.disable_fn=already_loaded_lib.modern_kimi_model_events_disable_r8669zq
        self.disable_fn.argtypes=[];self.disable_fn.restype=C.c_int
        self.selected=None
    def configure(self,label,ordinals):
        if not isinstance(label,str)or len(label)>128 or not 1<=len(ordinals)<=3 or ordinals!=sorted(set(ordinals))or any(type(x)is not int or not 0<=x<=8192 for x in ordinals):raise ValueError('finite selected RUN ordinals')
        q=json.loads(self.configure_fn(json.dumps({'label':label,'selected_forward_ordinals':ordinals}).encode()))
        if not q.get('configured'):raise RuntimeError(q)
        self.selected=list(ordinals);return q
    def read_after_original_reset_drain(self,expected_stage):
        q=json.loads(self.read_fn())
        if not q.get('all8_published')or set(q.get('owners',{}))!={str(i)for i in range(8)}:raise RuntimeError('not all owned records published after original drain')
        for key,v in q['owners'].items():
            if v.get('incomplete')or v['device']!=int(key)or v['PP_stage']!=expected_stage or not v['read_after_original_reset_drain']or v['max_events']!=640 or v['events_used']>640:raise RuntimeError('owner/event lifecycle mismatch')
            if [x['forward_ordinal']for x in v['calls']]!=self.selected:raise RuntimeError('sample coverage mismatch')
        if self.disable_fn()!=0:raise RuntimeError('pending records cannot be disabled')
        return q
