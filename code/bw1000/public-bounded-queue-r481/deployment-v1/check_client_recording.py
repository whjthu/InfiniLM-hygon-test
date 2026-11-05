"""Only new finiteclient main/protocol/parser/scorer; HTTP transport recording leaf."""
from pathlib import Path
import hashlib,importlib.util,json,sys,tempfile,time,types,itertools
D=Path(__file__).resolve().parent;G=D.parents[3];clock=G/'code/bw1000/public-token-clock-r480';sys.path.insert(0,str(clock));sys.path.insert(0,str(D));import finite_client as client
p=G/'code/bw1000/Deep-fixed-fullpre-experimental-profile-r454/source-v1/public_deep_model/protocol.py';spec=importlib.util.spec_from_file_location('public_deep_model.protocol',p);protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol);package=types.ModuleType('public_deep_model');package.__path__=[];sys.modules['public_deep_model']=package;sys.modules['public_deep_model.protocol']=protocol
from token_clock import PREFIX
H=lambda b:hashlib.sha256(b).hexdigest()
class Sock:
 def settimeout(self,n):assert n>0
class Response:
 status=200
 def __init__(self,body):self.body=body
 def read1(self,n):b=self.body;self.body=b'';return b
class Connection:
 counter=itertools.count()
 def __init__(self,*a,**kw):self.sock=Sock();self.rid='recording-'+str(next(self.counter))
 def request(self,method,path,body,headers):
  assert method=='POST'and headers['x-infinilm-token-clock']=='1';v=json.loads(body);text='#### 3'if v['suite']=='gsm8k'else'Answer: A';self.body='data: '+json.dumps({'id':self.rid,'choices':[{'delta':{'role':'assistant'}}]})+'\n\n'
  for i,t in enumerate([7,1]):self.body+=PREFIX+json.dumps(dict(request_id=self.rid,ordinal=i,token_id=t,visible=i==0,eos=i==1,special=i==1))+'\n\n'
  self.body+='data: '+json.dumps({'id':self.rid,'choices':[{'delta':{'content':text},'finish_reason':'stop'}]})+'\n\n'+PREFIX+json.dumps(dict(terminal=True,emitted=2,owned_finished=True,finish_reason='eos',observer_overflow=False))+'\n\ndata: [DONE]\n\n'
 def getresponse(self):return Response(self.body.encode())
 def close(self):pass
client.http.client.HTTPConnection=Connection
with tempfile.TemporaryDirectory()as tmp:
 root=Path(tmp);server=dict(run='recording',runtime_directory=str(root/'runtime'),actual_step='recording-step',source_manifest_SHA256='s',provider_config_SHA256='c',gpu_start_epoch=time.time(),Native_seconds=120);cfg=root/'server.json';cfg.write_text(json.dumps(server));run=protocol.Run(server)
 run.write('MODEL_READY.json',dict(Model_READY=True,Native_internal_owner_count=8,HC_mode=0,Native_SHA256='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1',fixed_profile=dict(fixed_mode=0,r366_fixed=0,HC_groups=86),queue_policy=dict(enabled=True)))
 records=[dict(case_id=str(i),gold=dict(id=str(i),suite='gsm8k'if i<2 else'mmlu',gold_numeric='3',gold_choice='A'))for i in range(4)]
 rows=[]
 for i in range(20):
  body={'suite':'gsm8k'if i%4<2 else'mmlu'};rows.append(dict(index=i,case_id=str(i%4),ordinal=[0,1,128,129][i%4],body=body,actual_body_SHA256=H(json.dumps(body,separators=(',',':')).encode()),prompt_IDs_SHA256='recording',arrival_offset_s=0))
 plan=root/'plan.json';plan.write_text(json.dumps(dict(records=records,requests=rows,ordinals=[0,1,128,129])))
 config=root/'client.json';config.write_text(json.dumps(dict(private_plan=str(plan),private_plan_SHA256=H(plan.read_bytes()),base_url='http://recording.invalid',output=str(root/'output'),pair='A',queue_policy=True,deadline_unix=time.time()+10,server_config=str(cfg),server_config_SHA256=H(cfg.read_bytes()))))
 client.main(config);q=json.loads((root/'output/result.json').read_bytes());assert q['completed']and q['sent']==q['terminal']==20 and q['quality_summary']['correct']==20 and q['quality_summary']['natural_EOS']==20 and q['missing']==0 and run.stop_requested()and all(r.get('server_request_id')for r in q['rows'])and len({r['server_request_id']for r in q['rows']})==20
 # Reuse exact same synthetic20/body/gold, change only schedule mode and outputnamespace.
 config2=json.loads(config.read_bytes());config2.update(pair='B',output=str(root/'output-B'));config.write_text(json.dumps(config2));client.main(config);qb=json.loads((root/'output-B/result.json').read_bytes());assert qb['completed']and len(qb['rows'])==20 and len(qb['waves'])==2 and qb['rows'][4]['planned_cancel']=='active_after_token'and qb['STOP_requested_epoch']
print(json.dumps(dict(passed=True,actual_finite_main=True,actual_original_Run_STOP=True,actual_parser_and_original_scorer=True,planned=20,HTTP_transport='RecordingMemoryLeaf',Native=False,GPU=False,actual_HTTP_external=False,actual_full945_data=False)))
