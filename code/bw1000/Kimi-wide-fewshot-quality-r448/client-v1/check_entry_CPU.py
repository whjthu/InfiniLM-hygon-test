"""Actual changedclient/controller/publicfactory composition; Native is recorded.

Does not encode, load, import Native or replay876/937. Real local HTTP257 calls
exercise the fixed256 main plus cancel boundary. Original576 report shapes are
private pinned recording templates, never new device/quality evidence.
"""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from unittest.mock import patch
from fractions import Fraction
from enum import Enum
import copy,hashlib,json,os,tempfile,threading,time
import client,controller,collect

class Reasons(str,Enum):
    EOS_TOKEN='eos_token';LENGTH='length';TIMEOUT='timeout';CANCELED='canceled';ERROR='error';STOP_STRING='stop_string'

def check(directory,manifest):
    D=Path(directory);pins=json.loads((D/'INPUT_PINS.json').read_bytes());host=controller.host
    enc=json.loads(host(pins['encoding_path']).read_bytes());cases=json.loads(host(pins['cases_path']).read_bytes())['cases']
    templates=json.loads((D/'CPU_TEMPLATE_PINS.json').read_bytes());originals=[]
    for pin in templates['reports']:
        raw=host(pin['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==pin['SHA256'];originals.append(json.loads(raw))
    from public_kimi_native.provider import KimiProvider
    from public_kimi_text.prepared import SelectedKimiBridge
    settings=dict(model_path='/shared/Kimi-K3',device='cuda',dtype='bfloat16',tensor_parallel_size=16,pipeline_parallel_size=2,pipeline_parallel_stage=0,cache_type='static',max_batch_size=1,max_cache_len=32768,max_tokens=4096,temperature=0.0,top_p=1.0,top_k=1,enable_graph=False,skip_load=False,enable_prefix_caching=False,attn_backend='default',use_mla=False,moe_ep_backend='disabled',moe_ep_size=1,weight_load_mode='async',kv_transfer_config=None,pre_transpose=False)
    outcomes=[]
    for scenario in ['complete256','cancel-after-first','foreign-claim','original-deadline']:
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);runtime=root/'results/public-Kimi-text-provider-r434';runtime.mkdir(parents=True)
            run='CPU448-'+scenario;step='CPU.recording';source=manifest['provider_source_SHA256'];config_sha=manifest['config_SHA256'];epoch=time.time()-(28150 if scenario=='original-deadline'else 0)
            config=dict(model_path='/shared/Kimi-K3',gpu_start_epoch=epoch,Native_seconds=28800,service_seconds=28200)
            reports=[];rows=[];error=[];server=None;thread=None
            keep=['node','TP','PP','layout','PP_ranges','cache_geometry','manifest_SHA256','preload_memory_gate','owner_affinity','library_sha256','model_weight_keys_validated','actual_allstage_weight_keys']
            for node,t in enumerate(originals):
                r={k:copy.deepcopy(t[k])for k in keep};r.update(run=run,node=node,public_source_manifest_SHA256=source,phase='weights_ready',completed=False,agent_requests=[],cleanup_errors=[])
                reports.append(r)
            def publish_reports():
                for node,r in enumerate(reports):client.save(runtime/('Kimi-public-r428-'+run+'-node%d.json'%node),r)
            def owner_leaf(c,prepared,stop,on_ready):
                try:
                    for node in range(4):client.save(runtime/('claim-'+run+'-node%d.json'%node),dict(run=run,actual_step=step,node=node,source_manifest_SHA256=source if scenario!='foreign-claim'or node else'FOREIGN',config_SHA256=config_sha,PID=os.getpid()))
                    publish_reports();on_ready()
                    if not stop.wait(120):raise TimeoutError('Bounded CPU recording owner stop')
                    for r in reports:r.update(completed=True,protocol_before_destroy_passed=True,all8_owned_workers_shutdown_completed=True,serving_protocol_complete=True,agent_native_forward_calls=sum(len(enc['cases'][i]['prefill_chunk_tokens'])+len(row['generated_ids'])-1 for i,row in enumerate(rows)))
                    publish_reports()
                except BaseException as e:error.append(repr(e));raise
            bridge=SelectedKimiBridge(None,32768,Reasons,default_max_tokens=4096)
            provider=KimiProvider(config,dict(engine=bridge,FinishReason=Reasons),owner=owner_leaf)
            assert provider(**settings)is provider
            provider.start();assert provider.is_healthy()and provider.ready.is_set()
            class Handler(BaseHTTPRequestHandler):
                def log_message(self,*a):pass
                def do_POST(self):
                    try:
                        i=len(rows);raw=self.rfile.read(int(self.headers['Content-Length']));body=json.loads(raw);e=enc['cases'][i];case=cases[i]
                        assert client.sha(raw)==e['request_body_SHA256']and body['max_tokens']==4096 and body['chat_template_kwargs']==dict(enable_thinking=False)
                        if case['suite']=='gsm8k':answer='#### '+str(Fraction(case['gold_numeric'])+(1 if i==3 else 0))
                        else:answer='Answer: '+case['gold_choice']
                        if i==2:answer='recorded unparseable content'
                        natural=i!=1;n=3 if natural else 4096;generated=[65]*(n-1)+([163586]if natural else[65]);now=time.monotonic();ident='CPU448-recorded-'+str(i)
                        row=dict(request_id=ident,input_tokens=e['prompt_tokens'],output_budget=4096,prefill_mode='W512',normal_reset=True,finish='eos'if natural else'length',history_render_audit=copy.deepcopy(e['history_render_audit']),generated_ids=generated,started_monotonic=now,finished_monotonic=now+n*.000001,backend_token_monotonic=[now+(j+1)*.000001 for j in range(n)],long_prefill_observations=[dict(full_prompt_consumed=True,input_tokens_committed=e['prompt_tokens'],actual_forward_records=len(e['prefill_chunk_tokens']),width8_prefill_verify_commit_count=0)])
                        rows.append(row)
                        for r in reports:r['agent_requests']=copy.deepcopy(rows)
                        publish_reports()
                        if scenario=='cancel-after-first':client.save(runtime/('STOP-'+run+'.json'),dict(identity=dict(run=run,source_manifest_SHA256=source,config_SHA256=config_sha,gpu_start_epoch=epoch),action='stop'))
                        wire=dict(id=ident,choices=[dict(index=0,message=dict(role='assistant',content=answer),finish_reason='stop'if natural else'length')],usage=dict(prompt_tokens=e['prompt_tokens'],completion_tokens=n,total_tokens=e['prompt_tokens']+n))
                        payload=json.dumps(wire).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
                    except BaseException as e:error.append(repr(e));self.send_error(500)
            try:
                server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever);thread.start()
                argv=['--base-url','http://127.0.0.1:%d/v1'%server.server_port,'--runtime-dir',str(runtime),'--run',run,'--expected-step',step,'--source-sha',source,'--config-sha',config_sha,'--model-sha',originals[0]['manifest_SHA256'],'--cases',str(host(pins['cases_path'])),'--requests',str(host(pins['requests_path'])),'--encoding-receipt',str(host(pins['encoding_path'])),'--output-dir',str(root/'client'),'--deadline-unix',str(epoch+28200),'--gpu-start-epoch',str(epoch)]
                rc=client.main(argv);q=json.loads((root/'client/result.json').read_bytes())
                provider.stop();assert provider.closed and provider.thread is not None and not provider.thread.is_alive()and not error
                assert thread.is_alive(), 'Native provider stop must not join/own the public HTTP thread'
                assert not any('uvicorn'in t.name.lower()or'http'in t.name.lower()and t is provider.thread for t in threading.enumerate())
                if scenario=='complete256':
                    assert rc==0 and q['execution_passed']and q['completed_HTTP']==256 and q['first_public_HTTP_success']['index']==0
                    assert not q['rows'][1]['correct']and q['rows'][1]['cap_hit']and q['rows'][2]['parse_failure']and not q['rows'][3]['answer_matches_gold']
                    # The actual terminalcollector consumes original real schema,
                    # with Native/owner/memory fields explicitly recording leaves.
                    p=root/'candidates/Kimi-wide-fewshot-quality-r448-CPU-v1/encoding.json';p.parent.mkdir(parents=True);p.write_bytes(host(pins['encoding_path']).read_bytes())
                    terminal=collect.collect(root,run,step,source,root/'client/result.json',time.time()+15)
                    assert terminal['normal32destroy']and terminal['external_HTTP']==256 and not terminal['all_correct_gate']
                elif scenario=='cancel-after-first':assert rc==1 and q['completed_HTTP']==q['attempted_HTTP']==1 and q['planned_HTTP']==256 and q['rows'][0]['correct']
                else:assert rc==1 and q['attempted_HTTP']==0 and q['planned_HTTP']==256
                outcomes.append(dict(scenario=scenario,actual_client_main=True,actual_local_HTTP=len(rows),client_RC=rc,completed=q['completed_HTTP'],planned=256,actual_publicfactory_start_stop=True,actual_shared_ready_handoff=True,original_epoch_maintained=True,actual_passive_PID_owned_LISTEN=scenario!='foreign-claim',recording_Native=True,actual_Native=False))
            finally:
                if not provider.closed:provider.stop()
                if server:server.shutdown();server.server_close()
                if thread:thread.join(timeout=2);assert not thread.is_alive()
    return dict(passed=True,scenarios=outcomes,actual_whole_newclient=True,actual_complete_fixed256_recorded_Native_HTTP=True,CPU937_encoding_repeated=False,recording_leaves=['Native generation and whole4node constructor/weights/finite/destroy reports from exact old576 schema','Public HTTP response producer; actual client urllib/parser and real KimiProvider lifetime run','FinishReason and codec-unused bridge lifecycle only; original937 actual codec retained'],Model=False,GPU=False,Native_dlopen=False,ASGI_or_Model_executed=False)
