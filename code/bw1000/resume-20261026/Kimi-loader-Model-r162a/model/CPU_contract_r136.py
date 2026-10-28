import importlib.util,pathlib,os,json,sys
D=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(D/'runtime'));spec=importlib.util.spec_from_file_location('observer',D/'runtime/model_event_observer_r135.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);os.environ['INFINILM_KIMI_MODEL_EVENTS_R8669ZQ']='1'
class F:
 def __init__(self,fn):self.fn=fn
 def __call__(self,*a):return self.fn(*a)
class Lib:
 def __init__(self):
  self.epoch=0;self.configs=[];self.reads=0;self.modern_kimi_model_events_configure_r8669zq=F(self.configure);self.modern_kimi_model_events_read_r8669zq=F(self.read);self.modern_kimi_model_events_disable_r8669zq=F(lambda:0)
 def configure(self,b):self.configs.append(json.loads(b));self.epoch+=1;return json.dumps({'configured':True,'epoch':self.epoch,'max_calls':3}).encode()
 def read(self):self.reads+=1;return json.dumps({'all8_published':True,'epoch':self.epoch,'owners':{str(i):{'PP_stage':0,'global_TP_rank':i,'calls':[{'forward_ordinal':0,'whole_task_stream_span_ms':5.,'original_tail_drain_ms':.2,'points':[{'name':'decoder.entry','layer':1,'is_KDA':True,'is_MoE':True},{'from':'decoder.entry','name':'attention.end','layer':1,'is_KDA':True,'is_MoE':True,'stream_span_ms':2.},{'from':'FFN.begin','name':'FFN.end','layer':1,'is_KDA':True,'is_MoE':True,'stream_span_ms':2.}]}],'read_after_original_reset_drain':True}for i in range(8)}}).encode()
class Control:
 def collect(self,label,v):return[v.copy()for _ in range(4)]
l=Lib();rows=[];p=m.ModelEvents(l,Control(),0,rows,True);p.set_next('on',True);p.after_original_reset()
for i in range(3):p.before_forward('verify',list(range(8)),[100]*8,[8]*8,i)
assert len(l.configs)==1 and l.reads==0 and p.current['actual_forward_map'][0]['actual_verify_index_zero_based']==2
p.set_next('off',False);p.after_original_reset();assert len(rows)==1 and l.reads==1
p.before_forward('verify',[1],[100],[8],7);p.after_normal_destroy();assert len(l.configs)==1 and len(rows)==2
p.set_next('earlyEOS',True);p.after_original_reset();p.before_forward('verify',[1],[100],[8],8);p.after_normal_destroy();assert 'UNKNOWN'in rows[-1]['coverage']and len(l.configs)==1
assert m.overhead(1.,1.02,1.)['within_3pct_diagnostic_gate']and not m.overhead(1.,1.04,1.)['within_3pct_diagnostic_gate']
print(json.dumps({'passed':True,'CPU_mock_only':True,'GPU_qualified':False,'cases':['no early read','warm2 then actual T64 epoch0','4node schedule equality','originalreset flush8owners','OFF no configure','earlyEOS honestUNKNOWN','3pct perturbationgate']}))
