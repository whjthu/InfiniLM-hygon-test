"""Actual adapter calls with a recording CABI boundary; never loads a Native or GPU."""
import ctypes as C,json,types,threading
from long_capture_native import Native,declare_long_capture,service_geometry,committed_feature_prefix

def reject(fn):
 try:fn()
 except (ValueError,RuntimeError,AssertionError):return
 raise AssertionError('Invalid CPU contract accepted')
class Fn:
 def __init__(self,callback):self.callback=callback;self.calls=[]
 def __call__(self,*args):self.calls.append(args);return self.callback(*args)
class Events:
 def before_forward(self,*args):pass
 def after_forward(self,*args):pass
 def after_original_reset(self):pass
def run():
 assert service_geometry(2,2,1)=={'target_collective_cards':32,'external_draft_cards':1,'all_service_cards':33}
 assert service_geometry(2,2,0)['all_service_cards']==32
 for values in [(4,2,1),(2,1,1),(2,2,2),(2,2,True),('2',2,1)]:reject(lambda:service_geometry(*values))
 reject(lambda:Native({},enabled=False));reject(lambda:Native({},enabled=True,draft_cards=0))
 calls=[];failure=[False]
 def begin(ctx,epoch,on,count,logical,past,width):
  assert isinstance(ctx,C.c_void_p)
  if on:
   cast=lambda x:C.cast(x,C.POINTER(C.c_int32))[0];calls.append((epoch,on,count,cast(logical),cast(past),cast(width)))
  else:assert count==0 and logical is past is width is None;calls.append((epoch,on,count))
  return 1 if failure[0]else 0
 lib=types.SimpleNamespace(modern_kimi_long_features_capabilities_r305=Fn(lambda:0x01320080),modern_kimi_long_features_begin_r305=Fn(begin),modern_kimi_pair_append_true_input_r247=Fn(lambda *args:0),modern_kimi_engine_error_r2475=lambda:b'CPU boundary failure',modern_kimi_batch_mode_r9122=Fn(lambda *args:0),modern_kimi_batch_reset_r9122=Fn(lambda *args:0))
 native=object.__new__(Native);native._adapter_owner=threading.get_ident();native.lib=lib;native.ctx=C.c_void_p(123);native.epoch=0;native.vocab=163840;native.serial=0;native.input_phase='prompt';native.state_mode=False;native.model_events=Events();native.long_begin,native.append_function=declare_long_capture(lib)
 assert lib.modern_kimi_long_features_begin_r305.argtypes==[C.c_void_p,C.c_uint64,C.c_int,C.c_size_t,C.c_void_p,C.c_void_p,C.c_void_p]
 native.arm([1],[32760],[8],True);assert calls[-1]==(1,1,1,1,32760,8)and native.epoch==1
 before=len(calls)
 foreign=[]
 def wrong_owner():
  try:native.arm([],[],[],False)
  except AssertionError:foreign.append(True)
 thread=threading.Thread(target=wrong_owner);thread.start();thread.join();assert foreign==[True]and len(calls)==before
 for args in [([1],[32760],[9],True),([1],[0],[513],True),([True],[0],[1],True),([1],[True],[1],True),([1],[32768],[1],True),([1,2],[0,0],[1,1],True)]:reject(lambda:native.arm(*args))
 assert len(calls)==before and native.epoch==1
 failure[0]=True;reject(lambda:native.arm([1],[1000],[8],True));assert native.epoch==1;failure[0]=False
 native.arm([],[],[],False);assert calls[-1]==(2,0,0)and native.epoch==2
 records=[]
 def finish(kind,ids,past,width,capture,logits,out,flags,wall,callback):
  if callback:callback([7])
  record={'kind':kind,'capture':capture,'past':past,'width':width};records.append(record);return [7],({1:b'CPUcapture'}if capture else None),record
 native.finish=finish
 native.call_append(1,512,[7]*512,capture=True);assert calls[-1][1:]==(1,1,1,512,512) and records[-1]['feature_inputs_committed']is True and records[-1]['capture']is True
 emitted=[];native.call_append(1,1024,[7],capture=True,on_selected=lambda selected:emitted.append(selected));assert emitted==[[7]]
 before=len(native.append_function.calls);native.input_phase='decode';reject(lambda:native.call_append(1,1024,[7],capture=True));assert len(native.append_function.calls)==before
 native.input_phase='prompt';reject(lambda:native.call('verify',[1],[8192],[[7]],True));ledger=[[1,2,8192,1]];native.rows=lambda ids:[row[:]for row in ledger];native.seal_prompt(8192,speculate=True);assert native.input_phase=='decode'and native.state_mode and len(lib.modern_kimi_batch_mode_r9122.calls)==1 and ledger==[[1,2,8192,1]]
 reject(lambda:native.call('prefill',[1],[0],[[7]],True));reject(lambda:native.call('verify',[1],[8192],[[7]*9],True));reject(lambda:native.call('verify',[1],[32768],[[7]],True))
 raw=bytes([1])*3*35840*2+bytes([2])*5*35840*2
 assert committed_feature_prefix(raw,8,3,committed=True)==bytes([1])*3*35840*2
 reject(lambda:committed_feature_prefix(raw,8,3,committed=False));reject(lambda:committed_feature_prefix(raw,8,9,committed=True));reject(lambda:committed_feature_prefix(raw[:-1],8,3,committed=True))
 bad=types.SimpleNamespace(**vars(lib));bad.modern_kimi_long_features_capabilities_r305=Fn(lambda:0x01200500);reject(lambda:declare_long_capture(bad))
 reject(lambda:native.reset(True));native.reset(False);assert native.input_phase=='prompt'and not native.state_mode and len(lib.modern_kimi_batch_reset_r9122.calls)==1
 return {'completed':True,'passed':True,'CPU_only':True,'GPU':0,'Native_dlopen':False,'HTTP':0,'actual_adapter_methods_executed':True,'recording_CABI_boundary_only':True,'checks':['explicit32targetplus1draftcost','CABItypesandcapabilitycookie','foreignownerrejectedbeforedisabledclearCABI','actuallongbeginroutingandepochonfailure','invalidmetadatarejectedbeforeCABI','append512captureandcommittedpublication','sealwithoutresetpreservesrowledger','unsealed/oversizeverifyrejected','rejectedverifyfeatureprefixnotpublished'],'limitations':'No actual CAPI execution, Context owner validation, device capture, worker transport, target quality or performance proof.'}
if __name__=='__main__':print(json.dumps(run(),indent=2))
