"""Changed six-request actual HTTP client, recorded Native/owner leaves only."""
from pathlib import Path
import argparse,hashlib,json,os,tempfile,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from unittest.mock import patch
import client,controller,workload


def run(fixture,encoded):
 D=Path(__file__).resolve().parent
 card=json.loads((D/'FIXTURE_CARD.json').read_bytes())
 private=json.loads(Path(fixture).read_bytes())['sequences'][1]
 encoding=json.loads(Path(encoded).with_name('CPU-input-encoding.json').read_bytes())['cases'][1]
 assert [v['delivery']for v in card['plan']]==['P0','P1','P0','P1','P1','P0']
 seen=[]
 with tempfile.TemporaryDirectory()as tmp:
  root=Path(tmp);runtime=root/'runtime';runtime.mkdir()
  original=private['body']
  f=card['plan'][0]
  # Exact already-read local private sources; nothing is transmitted by this test.
  p=root/Path(f['source_path']).relative_to('/refactor');p.parent.mkdir(parents=True);p.write_bytes(Path(encoded).parent.joinpath('source-request-1.json').read_bytes())
  assert hashlib.sha256(p.read_bytes()).hexdigest()==f['source_SHA256']
  p=root/Path(f['private_encoded_case']).relative_to('/refactor');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(encoded).read_bytes())
  run='local-six-SSE';source='CPU-provider-source';config='CPU-config';step='CPU.recording'
  (runtime/('claim-'+run+'-node0.json')).write_text(json.dumps(dict(run=run,node=0,actual_step=step,source_manifest_SHA256=source,config_SHA256=config,PID=os.getpid())))
  report=dict(public_source_manifest_SHA256=source,library_sha256='8df87f0741b13c90f658174a793f0068efb4eb8097eddf1d5f44edb4d208e1a2',agent_requests=[])
  source_report=runtime/('Kimi-public-r428-'+run+'-node0.json')
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*args):pass
   def do_POST(self):
    i=len(seen);assert i<6;arm=card['plan'][i]
    raw=self.rfile.read(int(self.headers['Content-Length']));body=json.loads(raw)
    assert hashlib.sha256(raw).hexdigest()==arm['actual_request_SHA256']
    assert body['text_delivery']==arm['delivery']
    assert {k:v for k,v in body.items()if k!='text_delivery'}==original
    seen.append([i,body['text_delivery']])
    gold=arm['gold'];args=dict(path=gold['path'],content=json.dumps(gold['content']))
    call=dict(id='call_cpu_new_'+str(i),type='function',function=dict(name=gold['function'],arguments=json.dumps(args)))
    ident='CPU-public-'+str(i);now=time.monotonic()
    # Different output lengths exercise per-request accounting, not cross-arm equality.
    count=3+i;generated=[65]*(count-1)+[163586]
    usage=dict(prompt_tokens=9613,completion_tokens=count,total_tokens=9613+count)
    report['agent_requests'].append(dict(request_id=ident,prefill_mode='W512',normal_reset=True,finish='eos',input_tokens=9613,output_budget=8192,history_render_audit=encoding['history_render_audit'],generated_ids=generated,cleanup_errors=[],long_prefill_observations=[dict(full_prompt_consumed=True,input_tokens_committed=9613)],backend_token_monotonic=[now+.001*(j+1)for j in range(count)],started_monotonic=now))
    source_report.write_text(json.dumps(report))
    self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
    events=[dict(id=ident,choices=[dict(index=0,delta=dict(reasoning_content='recorded reasoning'),finish_reason=None)]),dict(id=ident,choices=[dict(index=0,delta=dict(content='recorded content'),finish_reason=None)])]
    # Real parser must concatenate tool name/arguments split across SSE frames.
    argument=call['function']['arguments'];middle=len(argument)//2
    events += [dict(id=ident,choices=[dict(index=0,delta=dict(tool_calls=[dict(index=0,id=call['id'],type='function',function=dict(name=gold['function'],arguments=argument[:middle]))]),finish_reason=None)]),dict(id=ident,choices=[dict(index=0,delta=dict(tool_calls=[dict(index=0,function=dict(arguments=argument[middle:]))]),finish_reason=None)]),dict(id=ident,choices=[dict(index=0,delta={},finish_reason='tool_calls')]),dict(id=ident,choices=[],usage=usage)]
    for event in events:self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode());self.wfile.flush()
    self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
  server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever);thread.start()
  try:
   with patch.object(workload,'R',root),patch.object(client,'wait_listener',lambda *args:None):
    rc=client.main(['--base-url','http://127.0.0.1:%d/v1'%server.server_port,'--runtime-dir',str(runtime),'--run',run,'--expected-step',step,'--source-sha',source,'--config-sha',config,'--fixtures',str(D/'FIXTURE_CARD.json'),'--output-dir',str(root/'output'),'--deadline-unix',str(time.time()+180)])
   q=json.loads((root/'output/result.json').read_bytes());assert rc==0 and q['passed'],q
   assert seen==[[i,v]for i,v in enumerate(['P0','P1','P0','P1','P1','P0'])]
   assert q['attempted_HTTP']==q['completed_HTTP']==len(q['rows'])==6
   assert [v['warm']for v in q['rows']]==[True,True,False,False,False,False]
   assert [v['usage']['completion_tokens']for v in q['rows']]==list(range(3,9))
   assert len({v['request_id']for v in q['rows']})==6
   for row in q['rows']:
    clock=row['client_clock'];assert all(clock[k]is not None for k in ['first_payload_delta_s','first_reasoning_payload_s','first_content_payload_s','first_tool_payload_s'])
    assert len(clock['events'])==7 and len(clock['line_arrivals'])==14
    assert clock['chunks_are_not_tokens']and row['semantic_gold']and row['natural_EOS']
  finally:server.shutdown();server.server_close();thread.join(timeout=2)
 return dict(passed=True,actual_local_HTTP=6,actual_local_SSE=6,delivery_schedule=seen,warm_indices=[0,1],formal_indices=[2,3,4,5],variable_output_counts_recorded=list(range(3,9)),actual_client_main=True,actual_HTTP_parser_tool_argument_fragments=True,full_original_9613_body_preserved=True,controller_scope=controller.scope_contract(),recording_leaves=['Native generation, finite and owner report','PID-owned listener proof only'],codec_876_repeated=False,Native_dlopen=False,GPU=0,Slurm=0,Model=False)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--encoded',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 try:q=run(a.fixture,a.encoded)
 except BaseException as error:
  import traceback
  Path(a.output).write_text(json.dumps(dict(passed=False,error=repr(error),traceback=traceback.format_exc(),GPU=0,Slurm=0),indent=2)+'\n');raise
 Path(a.output).write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
