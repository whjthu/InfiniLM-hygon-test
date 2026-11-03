"""Changed two-request client uses real HTTP/SSE; Native/owner/listener leaves recorded."""
from pathlib import Path
import argparse
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import tempfile
import threading
import time
from unittest.mock import patch

import client
import workload


def run(sealed):
 D=Path(__file__).resolve().parent;seal=json.loads(Path(sealed).read_bytes());assert seal['read_only']and seal['new_jobs']==0
 card=json.loads((D/'FIXTURE_CARD.json').read_bytes());seen=[]
 with tempfile.TemporaryDirectory()as tmp:
  root=Path(tmp);runtime=root/'runtime';runtime.mkdir()
  for f in seal['files']:
   raw=f['rawtext'].encode();assert len(raw)==f['bytes']and hashlib.sha256(raw).hexdigest()==f['SHA256']
   relative=Path(f['path']).relative_to('/public/home/aclsylqidf/hygon-inference-refactor-20261014');p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
  # The two original source bodies are materialized from the already-read
  # private qualified archive, independently pinned by the same fixture card.
  original=json.loads((Path(sealed).parent/'Kimi428-qualified-two-fixtures-private.json').read_bytes())
  for f in original['files']:
   p=root/Path(f['path']).relative_to('/public/home/aclsylqidf/hygon-inference-refactor-20261014');p.parent.mkdir(parents=True,exist_ok=True);raw=f['rawtext'].encode();assert hashlib.sha256(raw).hexdigest()==f['SHA256'];p.write_bytes(raw)
  run='local-real-HTTP';source='CPU-provider-source';config='CPU-config';step='CPU.recording'
  (runtime/('claim-'+run+'-node0.json')).write_text(json.dumps(dict(run=run,node=0,actual_step=step,source_manifest_SHA256=source,config_SHA256=config,PID=os.getpid())))
  report=dict(public_source_manifest_SHA256=source,library_sha256='8df87f0741b13c90f658174a793f0068efb4eb8097eddf1d5f44edb4d208e1a2',agent_requests=[])
  source_report=runtime/('Kimi-public-r428-'+run+'-node0.json')
  encoding=json.loads(seal['files'][0]['rawtext'])
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*args):pass
   def do_POST(self):
    index=len(seen);assert index<2
    raw=self.rfile.read(int(self.headers['Content-Length']));body=json.loads(raw);f=card['plan'][index]
    assert hashlib.sha256(raw).hexdigest()==f['actual_request_SHA256'];seen.append(index)
    gold=f['gold'];args=gold['arguments']if gold['function']=='calculate'else dict(path=gold['path'],content=json.dumps(gold['content']))
    call=dict(id='new-opaque-CPU-'+str(index),type='function',function=dict(name=gold['function'],arguments=json.dumps(args)))
    usage=dict(prompt_tokens=f['current_prompt_tokens'],completion_tokens=3,total_tokens=f['current_prompt_tokens']+3)
    ident='CPU-public-'+str(index);now=time.monotonic()
    report['agent_requests'].append(dict(request_id=ident,prefill_mode='W512',normal_reset=True,finish='eos',input_tokens=f['current_prompt_tokens'],output_budget=f['output_budget'],history_render_audit=encoding['cases'][index]['history_render_audit'],generated_ids=[65,66,163585],generated_model_tokens_recorded_not_real=True,cleanup_errors=[],long_prefill_observations=[dict(full_prompt_consumed=True,input_tokens_committed=f['current_prompt_tokens'])],backend_token_monotonic=[now+.1,now+.2,now+.3],started_monotonic=now))
    source_report.write_text(json.dumps(report))
    self.send_response(200)
    if not f['stream']:
     self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(json.dumps(dict(id=ident,choices=[dict(index=0,message=dict(role='assistant',tool_calls=[call]),finish_reason='tool_calls')],usage=usage)).encode())
    else:
     self.send_header('Content-Type','text/event-stream');self.end_headers()
     part=dict(index=0,**call)
     events=[dict(id=ident,choices=[dict(index=0,delta=dict(tool_calls=[part]),finish_reason=None)]),dict(id=ident,choices=[dict(index=0,delta={},finish_reason='tool_calls')]),dict(id=ident,choices=[],usage=usage)]
     for event in events:self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode())
     self.wfile.write(b'data: [DONE]\n\n')
  server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever);thread.start()
  try:
   with patch.object(workload,'R',root),patch.object(client,'wait_listener',lambda *args:None):
    rc=client.main(['--base-url','http://127.0.0.1:%d/v1'%server.server_port,'--runtime-dir',str(runtime),'--run',run,'--expected-step',step,'--source-sha',source,'--config-sha',config,'--fixtures',str(D/'FIXTURE_CARD.json'),'--output-dir',str(root/'output'),'--deadline-unix',str(time.time()+180)])
   result=json.loads((root/'output/result.json').read_bytes());assert rc==0 and result['passed']and seen==[0,1]
   assert result['attempted_HTTP']==result['completed_HTTP']==2 and len(result['rows'])==2
   assert [r['stream']for r in result['rows']]==[False,True]
  finally:server.shutdown();server.server_close();thread.join(timeout=2)
 return dict(passed=True,actual_local_HTTP=2,actual_local_HTTP_streams=[False,True],real_private_actual_CPU_inputs=True,
  recording_leaves=['Native generated response/owner report', 'PID-owned passive listener proof'],
  checkpoint_tokenizer_repeated=False,Native_or_SDK=False,GPU=False,Slurm_submitted=0,Model_result=False,
  qualification='Actual final client main/stdlibHTTP/SSE parsing/bodyhash/toolgold/opaqueID/owner schema handoff only; recorded outputs are not quality or throughput evidence.')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sealed',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 try:q=run(a.sealed)
 except BaseException as error:
  import traceback
  a.output.write_text(json.dumps(dict(passed=False,error=repr(error),traceback=traceback.format_exc(),Slurm_submitted=0,GPU=False),indent=2)+'\n');raise
 a.output.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
