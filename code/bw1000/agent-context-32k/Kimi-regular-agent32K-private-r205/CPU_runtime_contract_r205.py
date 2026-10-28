"""CPU transaction/selector fixture only. No model outputs or numeric GPU proof."""
import pathlib,sys,json,types
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent/'runtime'))
from long_ar_r179 import generate,generate_scoped_request,prefill_true_inputs
class Fixture:
 def __init__(self,fail_at=None):self.past=0;self.pending=None;self.seen=[];self.resets=0;self.fail_at=fail_at;self.verify_widths=[]
 def reset(self,state):self.resets+=1;self.past=0;self.pending=None
 def call(self,kind,ids,past,tokens,capture):
  assert ids==[1]and not capture and len(tokens)==1
  if kind=='prefill':assert past==[0]and self.past==0 and 1<=len(tokens[0])<=512;self.past=len(tokens[0]);self.seen.extend(tokens[0]);selected=[(tokens[0][-1]+1)%163840]
  else:
   assert past==[self.past]and self.pending is None and 1<=len(tokens[0])<=8
   if self.fail_at==self.past:raise ValueError('CPU metadata rejection fixture')
   self.pending=list(tokens[0]);self.verify_widths.append(len(self.pending));selected=[(x+1)%163840 for x in self.pending]
  return selected,None,{}
 def commit(self,ids,decisions):
  assert ids==[1]and self.pending is not None;d=decisions[0];assert d.accepted==len(self.pending)-1
  self.seen.extend(self.pending)
  if not d.terminal:self.past+=len(self.pending)
  self.pending=None
cases=[]
for n,cap in [(8192,4096),(17408,8192),(24576,8192)]:
 f=Fixture();prompt=[i%163840 for i in range(n)];emitted=[];generated,reason=generate(f,prompt,cap,set(),emitted.append,lambda step:False,100.,clock=lambda:1.)
 assert reason=='length'and generated==emitted and len(generated)==cap and generated[0]==(prompt[-1]+1)%163840
 assert f.seen[:n]==prompt and f.resets==0 and all(w<=8 for w in f.verify_widths)
 cases.append({'input':n,'output_budget':cap,'all_true_input_preserved':True,'first_emit_after_full_prompt':True,'in_prompt_resets':0})
f=Fixture();progress=[]
def on_progress(committed,total,transactions,wall):
 assert f.past==committed and f.pending is None and len(f.seen)==committed
 progress.append((committed,total,transactions))
f.input_commit_progress=on_progress
prefill_true_inputs(f,list(range(8192)),lambda step:False,100.,clock=lambda:1.)
assert progress[0]==(512,8192,0)and progress[-1]==(8192,8192,960)and len(progress)==16
f=Fixture(fail_at=520)
try:generate_scoped_request(f,list(range(8192)),4096,set(),lambda token:None,lambda step:False,100.,clock=lambda:1.)
except ValueError:pass
else:raise AssertionError('metadata fixture did not reject')
assert f.resets==2 and f.pending is None and f.past==0
f=Fixture();out=[];got,reason=generate_scoped_request(f,list(range(8192)),4096,set(),out.append,lambda step:step<0,100.,clock=lambda:1.);assert got==[]and reason=='canceled'and f.resets==2 and not out

from long_input_r179 import KimiLongBridge
from long_codec_r179 import KimiLongCodec
from xtml_codec_r143 import KimiXTMLCodec
from unittest.mock import patch
reasons=types.SimpleNamespace(ERROR='error',TIMEOUT='timeout',CANCELED='canceled')
sp=lambda cap:types.SimpleNamespace(max_tokens=cap,temperature=0.,stop=None)
for length,budget in [(8192,4096),(17408,8192),(24576,8192)]:
 e=KimiLongBridge(None,32768,reasons,encode_chat=lambda *args:('CPU admission fixture only',[1]*length))
 e.validate_data({'messages':[],'chat_template_kwargs':{'enable_thinking':True}},sp(budget))
 request=e.add_chat_request(messages=[],sampling_params=sp(budget),request_id='CPU-long-'+str(length))
 assert len(request.prompt_token_ids)==length and request.max_new_tokens==budget and not request.generated_token_ids
try:
 e=KimiLongBridge(None,32768,reasons,encode_chat=lambda *args:('CPU over-context fixture',[1]*24577));e.validate_data({'messages':[]},sp(8192))
except ValueError:pass
else:raise AssertionError('over-context accepted')
codec=KimiLongCodec.__new__(KimiLongCodec);observed=[];messages=[{'role':'assistant','reasoning_content':'full CPU thought fixture','content':'answer'}]
with patch.object(KimiXTMLCodec,'encode_chat',lambda self,msgs,kwargs,generation:(observed.append((msgs,kwargs))or('CPU no tokenizer',[]))):
 codec.encode_chat(messages,{'enable_thinking':True});assert observed[-1][0]is messages and observed[-1][1]['thinking_effort']=='high'
 codec.encode_chat(messages,{'enable_thinking':True,'reasoning_effort':'max'});assert observed[-1][1]['reasoning_effort']=='max'
 try:codec.encode_chat(messages,{'enable_thinking':True,'reasoning_effort':'unsupported'})
 except ValueError:pass
 else:raise AssertionError('unsupported effort silently ignored')
print(json.dumps({'passed':True,'CPU_only':True,'GPU':False,'fixture_not_model_outputs':True,'cases':cases,'normal_and_exception_cancellation_cleanup':True,'context32768_default4096_thinking8192_explicit_high':True,'full_history_and_thought_honored':True,'unsupported_effort_rejected':True,'numeric_KDA_MLA_equivalence':'UNPROVEN','GPU_GO':False}))
