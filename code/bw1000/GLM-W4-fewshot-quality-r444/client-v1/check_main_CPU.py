"""Necessary changed main: real HTTP/Run/CLI/files, recorded Model leaves.

Uses CPU918's original exact bodies and encoding without encoding again.
No old64 suite, tokenizer, Torch, Native, Slurm or weights are executed here.
"""
import argparse
import contextlib
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import traceback

D=Path(__file__).resolve().parent
sys.path.insert(0,str(D))
import quality_client as C


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def check(fixtures,encoding_path,output):
    report=dict(passed=False,completed=False,phases=[],errors=[],actual_HTTP=0,
        recorded_leaves=['model response','Native owner result/logits/reset','loaded owner affinity/ready'],
        actual_processor_or_GPU_executed=False,Native_dlopen=False,Model=False,Slurm=0,
        original_encoding_SHA256=C.digest(Path(encoding_path).read_bytes()),
        original_fixture_cases_SHA256=C.digest((Path(fixtures)/'cases.json').read_bytes()))
    primary=None
    def persist():C.save(output,report)
    persist()
    try:
        report['actual_GPU_hiding_env']={k:os.environ.get(k)for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']}
        assert all(v=='-1'for v in report['actual_GPU_hiding_env'].values())
        assert report['original_encoding_SHA256']==C.ENCODING and report['original_fixture_cases_SHA256']==C.DATA
        provider=D.parent/'execution-v1';provider_sha=C.digest((provider/'SOURCE_MANIFEST.json').read_bytes())
        source=provider/'public_glm_native'
        protocol=load('actual444_cpu_run',source/'protocol.py')
        execute=load('actual444_cpu_execute',source/'glm_prefill_execute_r236.py').execute_plan
        observed=load('actual444_cpu_observed',source/'prefill_window_audit.py').execute_observed
        enc={c['case_id']:c for c in C.read(encoding_path)['cases']}
        data=C.read(Path(fixtures)/'cases.json')['cases'];cases={c['id']:c for c in data}
        plan=C.read(Path(fixtures)/'requests-and-pairs.json');bodies={C.digest(C.compact(r['body'])):r for r in plan['requests']}
        chosen={r['case_id']for i,r in enumerate(plan['requests'])if(i//2)%6==0}
        schedule=[r for r in plan['comparison_schedule']if r['case_id']in chosen];assert len(schedule)==88
        with tempfile.TemporaryDirectory(prefix='glm444-main-contract-')as temp:
            T=Path(temp)
            for scenario in ['positive','foreign_ready','bad_counter','same_run_STOP','HTTP_error']:
                record=dict(name=scenario,passed=False,HTTP=0,error=None)
                report['phases'].append(record);persist()
                config=C.read(provider/'config.template.json')
                config.update(Native_seconds=28800,service_seconds=28200,runtime_directory=str(T/scenario),nodes=['CPU-recorded-stage0','CPU-recorded-stage1'])
                config_path=T/(scenario+'-config.json');C.save(config_path,config);config_sha=C.digest(config_path.read_bytes())
                epoch=time.time();identity=dict(run='r444-recorded-'+scenario,actual_step='CPU.changed-main',gpu_start_epoch=epoch,
                    source_manifest_SHA256=provider_sha,provider_config_SHA256=config_sha)
                config.update(identity);run=protocol.Run(config)
                prepared=dict(upstream_commit=C.UPSTREAM,eos_token_ids=[154820,154827,154829])
                run.write('PREPARED.json',dict(prepared_identity=prepared))
                for s in range(2):
                    for r in range(8):
                        run.write('ready-%d-%d.json'%(s,r),dict(stage=s,rank=r,Native_SHA256=C.NATIVE,config_SHA256=config['config_SHA256'],
                            loaded=True,postload_4GiB=True,service_agreement=True,
                            affinity=dict(actual_owner_report=dict(owners=[dict(device=r,changes=1,verified_mask=','.join(str(c)for c in range(4*r,4*r+4)))]))))
                run.write('MODEL_READY.json',dict(Model_READY=True,Native_constructed_loaded_by_provider=True,actual_owner_count=16,
                    TP=8,PP=2,rows=1,capacity=32768,Native_SHA256=C.NATIVE,prepared_identity=prepared,all16_loaded_postload4GiB_and_service_agreement=True))
                if scenario=='foreign_ready':
                    ready=C.read(run.path/'MODEL_READY.json');ready['identity']['actual_step']='wrong-step';C.save(run.path/'MODEL_READY.json',ready)
                owners={}
                for s in range(2):
                    for r in range(8):
                        q=dict(run=config['run'],actual_step=config['actual_step'],Native_SHA256=C.NATIVE,
                            source_manifest_SHA256=provider_sha,service_profile_manifest_SHA256=provider_sha,
                            TP=8,PP=2,all_serving_cards=16,cache_rows=1,cache_capacity=32768,service_requests=[])
                        owners[s,r]=q;C.save(run.path/('owner-%d-%d.json'%(s,r)),q)
                received=[];server_errors=[]
                class Handler(BaseHTTPRequestHandler):
                    def log_message(self,*args):pass
                    def do_POST(self):
                        try:
                            assert self.path=='/v1/chat/completions'
                            raw=self.rfile.read(int(self.headers['Content-Length']));request=bodies[C.digest(raw)];i=len(received);item=schedule[i]
                            assert item['case_id']==request['case_id'] and raw==C.compact(request['body'])
                            w=item['window'];assert self.headers['X-InfiniLM-Native-Profile']=='glm-nov16-w'+str(w)
                            received.append((request['case_id'],w,C.digest(raw)));record['HTTP']=len(received);persist()
                            if scenario=='HTTP_error'and i==3:self.send_error(503);return
                            entry=enc[item['case_id']];case=cases[item['case_id']]
                            cap=i==len(schedule)-1;ids=[21]*4096 if cap else[21,22,154820];ticks=[10+j*.2 for j in range(len(ids))]
                            trace=[];input_plan=[];past=0
                            for n in entry['prefill_chunk_tokens']:input_plan.append((past,[0]*n));past+=n
                            observed(input_plan,0,prefill=True,window=w,forward=lambda *a:None,vote=lambda:False,
                                sample=lambda:21,discard=lambda:None,record=trace,executor=execute)
                            if scenario=='bad_counter':trace[0]['actual_committed_group_chunks']=[999]
                            calls=[dict(T=n,past=sum(entry['prefill_chunk_tokens'][:j]),prefill=True,CAPI_returned=True,CAPI_RC=0,finite=True,Native_CAPI_s=.001)for j,n in enumerate(entry['prefill_chunk_tokens'])]
                            calls +=[dict(T=1,past=entry['prompt_tokens']+j,prefill=False,CAPI_returned=True,CAPI_RC=0,finite=True,Native_CAPI_s=.001)for j in range(len(ids)-1)]
                            response_id='recorded-response-'+str(i)
                            row=dict(request_id=response_id,request_index=i,admitted_index=i,sequence=i+1,
                                native_profile='glm-nov16-w'+str(w),prefill_control_window=w,service_profile_manifest_SHA256=provider_sha,
                                normal_reset_before=True,normal_reset_after=True,original_reset_all16_completed=True,
                                admission=dict(accepted_monotonic=9.,Native_prompt_token_ids_SHA256=entry['prompt_IDs_SHA256'],
                                    history_render_audit=dict(qualified=True,prequeue_bound=True,markers_sent_to_Native=False,prompt_tokens=entry['prompt_tokens'])),
                                max_new_tokens=4096,requested_output_budget=4096,prompt_tokens=entry['prompt_tokens'],actual_prompt_token_ids_SHA256=entry['prompt_IDs_SHA256'],
                                Native_continuation=[dict(kind='prefill',tokens=n)for n in entry['prefill_chunk_tokens']]+[dict(kind='decode',tokens=1)for _ in ids[1:]],
                                native_calls=calls,prefill_window_audit=trace,generated_token_ids=ids,generated_tokens=len(ids),actual_generated_tokens=len(ids),generated_token_ids_SHA256=C.digest(C.compact(ids)),
                                generated_natural_EOS=not cap,native_finish_reason='length'if cap else'eos_token',backend_token_monotonic=ticks,
                                started_monotonic=9.5,ended_monotonic=ticks[-1]+1,Native_forward_host_summary=dict(prefill_calls=len(input_plan),decode_calls=len(ids)-1),final_metadata=dict(recorded=True))
                            for s in range(2):
                                for r in range(8):
                                    ownerrow=copy.deepcopy(row)
                                    if s==0:
                                        for c in ownerrow['native_calls']:c['finite']=None
                                    owners[s,r]['service_requests'].append(ownerrow);C.save(run.path/('owner-%d-%d.json'%(s,r)),owners[s,r])
                            content='#### '+case['gold_numeric']if case['suite']=='gsm8k'else'Answer: '+case['gold_choice']
                            if i==1:content='#### -987654321'if case['suite']=='gsm8k'else'Answer: '+next(c for c in 'ABCD'if c!=case['gold_choice'])
                            if i==3:content='No parseable final answer'
                            reply=dict(id=response_id,choices=[dict(message=dict(role='assistant',content=content),finish_reason='length'if cap else'stop')],
                                usage=dict(prompt_tokens=entry['prompt_tokens'],completion_tokens=len(ids),total_tokens=entry['prompt_tokens']+len(ids)))
                            wire=C.compact(reply)
                            if scenario=='same_run_STOP'and i==3:run.stop()
                            self.send_response(200);self.send_header('Content-Length',str(len(wire)));self.end_headers();self.wfile.write(wire)
                        except BaseException as error:
                            server_errors.append(repr(error));self.send_error(500)
                server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever);thread.start()
                argv=['--base-url','http://127.0.0.1:'+str(server.server_port)+'/v1', '--provider-root',str(provider),
                    '--provider-source-sha',provider_sha,'--audited-owner-sha',C.digest((source/'native_owner.py').read_bytes()),
                    '--config',str(config_path),'--config-sha',config_sha,'--run',config['run'],'--expected-step',config['actual_step'],
                    '--service-report',str(run.path/'owner-1-0.json'),'--fixtures',str(fixtures),'--encoding-receipt',str(encoding_path),
                    '--output-dir',str(T/(scenario+'-client')),'--shards','6','--shard-index','0','--gpu-start-epoch',str(epoch),
                    '--deadline-unix',str(epoch+28200),'--timeout','20']
                try:
                    # Actual complete argparse/main; no replacement identity,
                    # row reader, protocol, executor, HTTP, or scoring function.
                    stdout=io.StringIO()
                    with contextlib.redirect_stdout(stdout):rc=C.main(argv)
                    result=C.read(T/(scenario+'-client')/'result.json')
                    record.update(client_returncode=rc,result_summary=result['summary'],actual_HTTP=result['actual_HTTP'],rows=len(result['rows']),
                        client_result_SHA256=C.digest((T/(scenario+'-client')/'result.json').read_bytes()),client_stdout=stdout.getvalue(),errors=result['errors'])
                    persist()  # preserve primary outcome before contract asserts
                    assert not server_errors,server_errors
                    assert result['planned_global_HTTP']==512 and result['planned_HTTP']==88
                    assert result['summary']['pooled']['expected_pairs']==44
                    if scenario=='positive':
                        assert rc==0 and result['execution_passed']and len(received)==len(result['rows'])==88
                        assert len(result['all16_owner_request_signatures'])==16
                        arms=result['summary']['pooled']['arms'];assert sum(v['caps']for v in arms.values())==1 and sum(v['parse_failures']for v in arms.values())==1
                        assert sum(v['wrong_or_invalid_observed']for v in arms.values())==3
                        assert result['summary']['pooled']['observed_complete_pairs']['complete_pairs']==44
                    else:
                        assert rc==1 and not result['execution_passed']
                        expected={'foreign_ready':(0,0),'bad_counter':(1,0),'same_run_STOP':(4,4),'HTTP_error':(4,3)}[scenario]
                        assert(result['actual_HTTP'],len(result['rows']))==expected
                        assert sum(v['uncompleted_or_not_attempted']for v in result['summary']['pooled']['arms'].values())==88-len(result['rows'])
                    record['passed']=True;persist()
                finally:
                    server.shutdown();server.server_close();thread.join(timeout=5);assert not thread.is_alive()
            # Actual copied final entry CLI in source-only mode for BOTH roles.
            sys.path.insert(0,str(provider))
            from public_glm_native.cli import main as cli
            cp=T/'actual-entry-config.json';config=C.read(provider/'config.template.json');config.update(Native_seconds=28800,service_seconds=28200)
            C.save(cp,config)
            cli_evidence=[]
            for stage in[0,1]:
                buf=io.StringIO()
                with contextlib.redirect_stdout(buf):cli(['--config',str(cp),'--config-sha',C.digest(cp.read_bytes()),'--source-sha',provider_sha,
                    '--stage',str(stage),'--run','cpu-source-only','--actual-step','CPU.changed-entry','--gpu-start-epoch',str(time.time())])
                q=json.loads(buf.getvalue());assert q['source_and_schema_valid']and not q['GPU']and not q['Native_dlopen'];cli_evidence.append(q)
        assert not any(n in sys.modules for n in ['torch','transformers','_infinilm'])
        report.update(passed=True,actual_HTTP=sum(p['actual_HTTP']for p in report['phases']),actual_whole_client_main=True,
            identity_not_mocked=True,actual_Run_and_all16_ready_consumed=True,actual_original_executor_observed_with_recording_Native_leaves=True,
            real_CPU918_encodings_reused_no_encode=True,quality_wrong_cap_parse_continue=True,partial_planned_denominators_kept=True,
            same_run_STOP_no_next_request=True,actual_private_entry_CLI_both_roles=cli_evidence,
            qualified_live_Model_or_W4_counter=False,old64_suite_replayed=False,provider_source_SHA256=provider_sha)
    except BaseException as error:
        primary=error;report['errors'].append(dict(error=repr(error),traceback=traceback.format_exc()))
    finally:
        report['completed']=True
        for name in ['torch','transformers','_infinilm']:report[name+'_imported']=name in sys.modules
        maps=Path('/proc/self/maps')
        report['project_Native_maps']=[line for line in maps.read_text().splitlines()if any(x in line for x in ['libGLM_V_on_demand_Model','libGLM_PP4_registry','_infinilm.cpython'])]if maps.exists()else None
        if report['project_Native_maps']:report['passed']=False
        persist()
    if primary:raise primary
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fixtures',required=True);p.add_argument('--encoding',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    q=check(a.fixtures,a.encoding,a.output);print(json.dumps(dict(passed=q['passed'],actual_HTTP=q['actual_HTTP'],output=a.output)))
