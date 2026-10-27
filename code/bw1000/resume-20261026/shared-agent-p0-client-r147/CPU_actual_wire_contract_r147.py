"""Replay actual Deep1145 SSE with strict original parser; CPU only.

Header dictionary casing is a controlled fixture: historical headers were not
recorded. Actual wire/hash/model tool call remain unchanged; no live HTTP pass.
"""
import pathlib,hashlib,importlib.util,json,tempfile,ast
D=pathlib.Path(__file__).resolve().parent;OLD=D.parent/'shared-agent-p0-client-r143/client_p0_r143.py'
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=load(OLD,'client_r143');new=load(D/'client_p0_r147.py','client_r147');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
wire=(D/'actual-Deep1145-request007.wire').read_bytes();assert hashlib.sha256(wire).hexdigest()=='85cac56bd69b1c6831eafe27750311bf95b64c4b7a3e6d744cc3a2377ee8997f'
def fn_ast(p,name):return ast.dump(next(n for n in ast.parse(p.read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name==name),include_attributes=False)
assert all(fn_ast(OLD,n)==fn_ast(D/'client_p0_r147.py',n)for n in ['sse_message','execute','workflow','ordinary','single_choice','negative','parallel_history','main'])
message,finish,usage=old.sse_message(wire);assert finish=='tool_calls'and usage=={'prompt_tokens':309,'completion_tokens':54,'total_tokens':363};assert len(message['tool_calls'])==1;call=message['tool_calls'][0];assert call['id']=='call_f08690b77251460ebdd0d1b4d5cd0127'and call['type']=='function'and call['function']['name']=='read_file'and json.loads(call['function']['arguments'])=={'path':'input.json'}
class Response:
 status=200
 def __init__(self,headers):self.headers=headers
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def read(self):return wire
rows=[];realopen=new.urllib.request.urlopen
with tempfile.TemporaryDirectory(prefix='actual-wire-header-')as tmp:
 try:
  for label,m,headers,expect in [('r143_lowercase_reproduces_failure',old,{'content-type':'text/event-stream; charset=utf-8'},False),('r147_lowercase_accepts',new,{'content-type':'text/event-stream; charset=utf-8'},True),('r147_mixedcase_accepts',new,{'Content-Type':'text/event-stream; charset=utf-8'},True),('r147_invalid_content_type_rejects',new,{'content-type':'application/json'},False)]:
   out=pathlib.Path(tmp)/label;out.mkdir();new.urllib.request.urlopen=lambda *args,**kwargs:Response(headers);accepted=False;error=None
   try:
    actual,reason,counts,e=m.WireClient('http://127.0.0.1:1/v1','fixture',out).request([{'role':'user','content':'fixture'}],stream=True);assert actual==message and reason==finish and counts==usage;accepted=True
   except AssertionError as exc:error=repr(exc)
   assert accepted==expect,label;rows.append({'name':label,'passed':True,'accepted':accepted,'expected_accepted':expect,'error':error})
 finally:new.urllib.request.urlopen=realopen
q={'CPU_only':True,'GPU':False,'actual_model_wire_replay':True,'live_HTTP_rerun':False,'full_SSE_workflow_qualified':False,'passed':True,'source_actual_step':'910374.1145','wire_SHA256':hashlib.sha256(wire).hexdigest(),'old_client_SHA256':sha(OLD),'new_client_SHA256':sha(D/'client_p0_r147.py'),'strict_parser_and_tool_execution_AST_unchanged':True,'actual_decoded_message':message,'actual_finish_reason':finish,'actual_usage':usage,'cases':rows,'historical_response_headers':'NOT_RECORDED; casing controlled in fixture. Exact failing-header byte cannot be asserted.','qualification':'Historical first SSE call was complete and parseable. Full four-step SSE workflow still requires new live HTTP/SSE run; no promotion of old failed case.'};(D/'CPU-actual-wire-contract-r147.json').write_bytes((json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps(q,ensure_ascii=False))
