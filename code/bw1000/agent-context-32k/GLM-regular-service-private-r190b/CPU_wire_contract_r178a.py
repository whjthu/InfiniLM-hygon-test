"""Pure CPU intercepted HTTP construction; not actual HTTP or model evidence."""
import json,pathlib,tempfile,unittest.mock
from client_GLM_regular_wire_r178a import WireClient
def validate():
 bodies=[]
 class Response:
  status=200;headers={'content-type':'application/json'}
  def __enter__(self):return self
  def __exit__(self,*x):pass
  def read(self):return json.dumps({'choices':[{'message':{'role':'assistant','content':'fixture only'},'finish_reason':'stop'}],'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}).encode()
 def urlopen(request,timeout):bodies.append(json.loads(request.data));return Response()
 messages=[{'role':'user','content':'complete history fixture'},{'role':'assistant','content':'','reasoning_content':'preserve thought'},{'role':'user','content':'next'}]
 with tempfile.TemporaryDirectory(prefix='glm-wire-r178a-')as tmp,unittest.mock.patch('urllib.request.urlopen',urlopen):
  c=WireClient('http://127.0.0.1:1/v1','GLM-5.3',pathlib.Path(tmp),4096)
  _,_,_,a=c.request(messages,max_tokens=4096,thinking=False)
  _,_,_,b=c.request(messages,max_tokens=8192,thinking=True)
 assert 'reasoning_effort'not in bodies[0]and bodies[1]['reasoning_effort']=='high'
 assert bodies[0]['max_tokens']==4096 and bodies[1]['max_tokens']==8192
 assert all(x['messages']==messages and x['chat_template_kwargs']['clear_thinking']is False for x in bodies)
 assert a['reasoning_policy']=='native_default_omitted'and b['reasoning_policy']=='high'
 return {'passed':True,'CPU_mock_only':True,'network_used':False,'GPU':False,'model_called':False,'native_default_4K':True,'explicit_high_8K':True,'full_messages_and_reasoning_retained':True}
if __name__=='__main__':print(json.dumps(validate()))
