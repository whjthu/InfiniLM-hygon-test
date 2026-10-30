"""CPU scheduling/failure tests only; fake CAPI, control and event publication."""
import copy,json,os,sys,types,unittest
from unittest.mock import patch
from model_event_observer_r255 import ModelEvents

class Function:
 def __init__(self,callback):self.callback=callback;self.calls=[]
 def __call__(self,*args):self.calls.append(args);return self.callback(*args)
class Library:
 def __init__(self):
  self.epoch=0
  self.raw=None
  self.modern_kimi_model_events_configure_r8669zq=Function(self.configure)
  self.modern_kimi_model_events_read_r8669zq=Function(lambda:json.dumps(self.raw).encode())
  self.modern_kimi_model_events_disable_r8669zq=Function(lambda:0)
 def configure(self,raw):
  value=json.loads(raw);assert value['selected_forward_ordinals']==[0]
  self.epoch+=1
  self.raw={'epoch':self.epoch,'all8_published':True,'owners':{str(i):{'calls':[{'forward_ordinal':0}],'read_after_original_reset_drain':True}for i in range(8)}}
  return json.dumps({'configured':True,'epoch':self.epoch,'max_calls':3}).encode()
class Control:
 def __init__(self):self.mismatch=False;self.maps=[]
 def collect(self,label,value):
  self.maps.append((label,copy.deepcopy(value)));rows=[copy.deepcopy(value)for _ in range(4)]
  if self.mismatch:rows[3]['T']=2
  return rows
class Contract(unittest.TestCase):
 def make(self,on=True):
  lib=Library();control=Control();records=[]
  with patch.dict(os.environ,{'INFINILM_KIMI_MODEL_EVENTS_R8669ZQ':'1'}):observer=ModelEvents(lib,control,0,records,enabled=on)
  observer.set_next('real-request',on);observer.after_original_reset()
  return observer,lib,control,records
 def call(self,observer,index,ids=(1,),widths=(1,),kind='verify'):
  observer.before_forward(kind,list(ids),[8635+index]*len(ids),list(widths),100+index)
 def warm(self,observer):
  self.call(observer,0);self.call(observer,1)
 def test_startup_optin_required(self):
  with patch.dict(os.environ,{},clear=True):
   with self.assertRaises(AssertionError):ModelEvents(Library(),Control(),0,[],enabled=True)
 def test_off_never_arms(self):
  o,l,c,r=self.make(False)
  for i in range(20):self.call(o,i)
  o.finish_after_original_drain('original_next_case_reset')
  self.assertEqual(len(l.modern_kimi_model_events_configure_r8669zq.calls),0)
  self.assertEqual(r[0]['coverage'],'profile_OFF')
 def test_prefill_and_wrong_shape_never_arm(self):
  o,l,c,r=self.make();self.warm(o)
  self.call(o,2,widths=(512,),kind='prefill')
  self.call(o,3,ids=(1,2,3,4,5,6,7,8),widths=(8,)*8)
  self.call(o,4,ids=(1,2),widths=(1,1));self.call(o,5,widths=(8,))
  self.assertFalse(l.modern_kimi_model_events_configure_r8669zq.calls)
 def test_actual_third_decode_one_finite_epoch(self):
  o,l,c,r=self.make();self.warm(o)
  self.assertFalse(l.modern_kimi_model_events_configure_r8669zq.calls)
  self.call(o,2)
  for i in range(3,50):self.call(o,i)
  self.assertEqual(len(l.modern_kimi_model_events_configure_r8669zq.calls),1)
  m=c.maps[0][1];self.assertEqual((m['C'],m['T'],m['actual_native_serial'],m['actual_verify_index_zero_based'],m['worker_RUN_ordinal_in_epoch']),(1,1,102,2,0))
  self.assertFalse(l.modern_kimi_model_events_read_r8669zq.calls)
 def test_early_EOS_does_not_schedule_future_RUN(self):
  o,l,c,r=self.make();self.warm(o);o.finish_after_original_drain('original_next_case_reset')
  self.assertFalse(l.modern_kimi_model_events_configure_r8669zq.calls)
  self.assertIn('UNKNOWN',r[0]['coverage']);self.assertNotIn('T64',r[0]['coverage'])
 def test_control_mismatch_fails(self):
  o,l,c,r=self.make();self.warm(o);c.mismatch=True
  with self.assertRaises(AssertionError):self.call(o,2)
  self.assertFalse(o.current['armed']);self.assertFalse(l.modern_kimi_model_events_read_r8669zq.calls)
 def test_incomplete_or_wrong_epoch_publication_fails(self):
  for defect in ['missing_owner','wrong_epoch','wrong_ordinal','not_drained']:
   with self.subTest(defect=defect):
    o,l,c,r=self.make();self.warm(o);self.call(o,2)
    if defect=='missing_owner':l.raw['owners'].pop('7')
    if defect=='wrong_epoch':l.raw['epoch']+=1
    if defect=='wrong_ordinal':l.raw['owners']['0']['calls'][0]['forward_ordinal']=1
    if defect=='not_drained':l.raw['owners']['0']['read_after_original_reset_drain']=False
    with self.assertRaises(AssertionError):o.finish_after_original_drain('original_next_case_reset')
    self.assertFalse(l.modern_kimi_model_events_disable_r8669zq.calls);self.assertFalse(r)
 def test_original_drain_publish_disable_then_next_case(self):
  o,l,c,r=self.make();self.warm(o);self.call(o,2)
  summary=types.ModuleType('summarize_decoder_events_r135');summary.summarize=lambda raw:{'CPU_fake_events':True}
  with patch.dict(sys.modules,{'summarize_decoder_events_r135':summary}):o.finish_after_original_drain('original_next_case_reset')
  self.assertEqual(len(r),1);self.assertIn('C1/T1',r[0]['coverage']);self.assertEqual(len(l.modern_kimi_model_events_disable_r8669zq.calls),1)
  o.set_next('next-real-request',True);o.after_original_reset();self.warm(o);self.call(o,2)
  self.assertEqual(l.epoch,2)
if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contract);result=unittest.TextTestRunner(verbosity=1).run(suite)
 print(json.dumps({'CPU_only':True,'fake_CAPI_control_and_events':True,'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'GPU_jobs':0,'HTTP':0}))
 sys.exit(0 if result.wasSuccessful()else 1)
