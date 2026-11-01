"""Explicit C1 adapter for305 capture. Target collectives stay32; service cost is33."""
import ctypes as C,os,time,threading
import numpy as np
from target_native_base import Native as TargetNative
P=C.c_void_p;I=C.c_int;S=C.c_size_t

def service_geometry(pp,tp_nodes,draft_cards):
 if type(pp)is not int or pp!=2 or type(tp_nodes)is not int or tp_nodes!=2 or type(draft_cards)is not int or draft_cards not in [0,1]:raise ValueError('Exact32target TP16PP2 plus0/1externaldraft required')
 return {'target_collective_cards':32,'external_draft_cards':draft_cards,'all_service_cards':32+draft_cards}

def declare_long_capture(lib):
 cap=lib.modern_kimi_long_features_capabilities_r305;cap.argtypes=[];cap.restype=C.c_uint64
 if cap()!=0x01320080:raise ValueError('Longcapture capability mismatch')
 begin=lib.modern_kimi_long_features_begin_r305;begin.argtypes=[P,C.c_uint64,I,S,P,P,P];begin.restype=I
 append=lib.modern_kimi_pair_append_true_input_r247;append.argtypes=[P,I,I,S,P,P,P,P,P];append.restype=I
 return begin,append

def committed_feature_prefix(raw,captured_width,accepted_inputs,*,committed):
 if committed is not True:raise ValueError('Targetcommitmustcompletebeforefeaturepublication')
 if type(captured_width)is not int or type(accepted_inputs)is not int or not 1<=accepted_inputs<=captured_width<=8:raise ValueError('Committeddecodewidth1..8only')
 if not isinstance(raw,bytes)or len(raw)!=captured_width*35840*2:raise ValueError('Exactcaptured5tapBF16bytesrequired')
 return raw[:accepted_inputs*35840*2]

class Native(TargetNative):
 def __init__(self,g,*,enabled=False,draft_cards=1):
  if enabled is not True or os.environ.get('INFINILM_KIMI_LONG_FEATURES_R305')!='1':raise RuntimeError('Longcapture adapter defaultOFF; process opt-in required')
  if draft_cards!=1 or type(draft_cards)is not int:raise ValueError('Explicit one external draft card required')
  service_geometry(g.get('pp',2),g.get('tp_nodes',2),draft_cards)
  super().__init__(g,draft_cards=draft_cards)
  self._adapter_owner=threading.get_ident()
  self.long_begin,self.append_function=declare_long_capture(self.lib);self.input_phase='new';self.state_mode=False
 def _owner(self):assert threading.get_ident()==self._adapter_owner,'ForeignlongcapturePythonowner'
 def arm(self,ids,past,width,on):
  self._owner()
  if type(on)is not bool:raise ValueError('Booleancaptureflagrequired')
  if on:
   if len(ids)!=1 or list(ids)!=[1]or type(ids[0])is not int or len(past)!=1 or len(width)!=1:raise ValueError('C1logical1captureonly')
   if type(past[0])is not int or type(width[0])is not int or past[0]<0 or not 1<=width[0]<=512 or past[0]>32768-width[0]:raise ValueError('Capturepast/width/context')
  next_epoch=self.epoch+1
  if not 1<=next_epoch<=2**63-1:raise ValueError('Captureepochexhausted')
  if on:
   arrays=[np.asarray(x,np.int32)for x in [ids,past,width]]
   self.ck(self.long_begin(self.ctx,next_epoch,1,1,*[self.ptr(x)for x in arrays]))
  else:self.ck(self.long_begin(self.ctx,next_epoch,0,0,None,None,None))
  self.epoch=next_epoch
 def reset(self,state):
  self._owner()
  if state is not False:raise ValueError('Longprefill starts with capturestateOFF; seal then mode switch withoutreset')
  self.arm([],[],[],False);super().reset(False);self.input_phase='prompt';self.state_mode=False
 def seal_prompt(self,expected_past,*,speculate):
  self._owner()
  if self.input_phase!='prompt' or type(expected_past)is not int or not 1<=expected_past<=32768 or type(speculate)is not bool:raise ValueError('Invalidpromptseal')
  before=self.rows([1]);assert before==[[1,2,expected_past,1]],'InitialC1active/shadow/committedpast required'
  self.ck(self.lib.modern_kimi_batch_mode_r9122(self.ctx,int(speculate)))
  after=self.rows([1]);assert after==before,'Modeswitchmustpreserveactualpastandrowledger'
  self.input_phase='decode';self.state_mode=speculate
 def call(self,kind,ids,past,tokens,capture,on_selected=None):
  self._owner()
  if ids!=[1]or type(ids[0])is not int or len(past)!=1 or type(past[0])is not int or past[0]<0 or len(tokens)!=1 or type(capture)is not bool:raise ValueError('ExactC1forwardmetadata')
  if not tokens[0]or past[0]>32768-len(tokens[0])or any(type(v)is not int or not 0<=v<self.vocab for v in tokens[0]):raise ValueError('ActualforwardIDs/context')
  if kind=='prefill':
   if self.input_phase!='prompt' or ids!=[1]or past!=[0]or len(tokens)!=1 or not 1<=len(tokens[0])<=512:raise ValueError('InitialC1promptseedonly')
  elif kind=='verify':
   if self.input_phase!='decode' or ids!=[1]or len(past)!=1 or len(tokens)!=1 or not 1<=len(tokens[0])<=8:raise ValueError('SealedC1decodewidth1..8')
  else:raise ValueError('Unsupportedlongcaptureforwardkind')
  result=super().call(kind,ids,past,tokens,capture,on_selected)
  result[-1].update(feature_capture_epoch=self.epoch if capture else None,feature_inputs_committed=kind=='prefill',verify_features_require_target_commit=kind=='verify')
  return result
 def call_append(self,logical,past,tokens,*,capture=False,on_selected=None):
  self._owner()
  if self.input_phase!='prompt' or self.state_mode:raise ValueError('Trueinputappend beforeverify/statecapture only')
  if type(logical)is not int or logical!=1 or type(past)is not int or past<1:raise ValueError('C1committedpositivepast')
  if not isinstance(tokens,(list,tuple))or not 1<=len(tokens)<=512 or past>32768-len(tokens)or type(capture)is not bool:raise ValueError('Trueinputwidth/context')
  if any(type(v)is not int or not 0<=v<self.vocab for v in tokens):raise ValueError('ActualvocabularyIDsrequired')
  widths=[len(tokens)];flat=np.asarray(tokens,np.int64);pos=np.arange(past,past+len(tokens),dtype=np.int64)
  logits=np.full((1,self.vocab),0x7fc0,np.uint16);out=np.full(1,-1,np.int64);flags=np.asarray([-1],np.int32)
  self.model_events.before_forward('prefill',[logical],[past],widths,self.serial);self.arm([logical],[past],widths,capture);tick=time.perf_counter()
  self.ck(self.append_function(self.ctx,logical,past,len(tokens),*[self.ptr(v)for v in [flat,pos,logits,out,flags]]))
  result=self.finish('append512',[logical],[past],widths,capture,logits,out,flags,time.perf_counter()-tick,on_selected)
  result[-1].update(feature_capture_epoch=self.epoch if capture else None,feature_inputs_committed=True,verify_features_require_target_commit=False)
  self.model_events.after_forward('prefill',result[-1]);return result
