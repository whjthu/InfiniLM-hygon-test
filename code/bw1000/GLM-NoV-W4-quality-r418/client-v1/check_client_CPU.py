"""Exercise the real client over HTTP plus freshly written owner/encoding files.

Only endpoint/Native/encoder leaves are recorded. No checkpoint, model,
tokenizer, Slurm, or GPU is executed; returned answers deliberately include
quality failures to verify they stay in denominators instead of aborting.
"""
import argparse
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace

D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('actual_quality_client',D/'quality_client.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)


def check():
    F=D.parent/'fixtures-r1'
    plan=C.read(F/'requests-and-pairs.json');dataset=C.read(F/'gsm8k-new64.json')
    bodies={C.digest(C.compact(r['body'])):r for r in plan['requests']}
    cases={r['id']:r for r in dataset['cases']}
    source='f'*64;profile='695b1a12bfd2929851b3ffa89a9267564f9fd9dc0045c4a9adfbcf7142b21584'
    received=[];server_errors=[]
    with tempfile.TemporaryDirectory(prefix='glm418-http-client-') as temp:
        T=Path(temp);run='CPU-recorded-owner'
        owner=T/('GLM-latest-full-owner-r290-'+run+'-stage1-rank0.json')
        encoded=[]
        for i,r in enumerate(plan['requests']):
            encoded.append(dict(case_id=r['case_id'],request_body_SHA256=r['compact_body_SHA256'],
                prompt_tokens=1,prompt_IDs_SHA256=C.digest(C.compact([500+i]))))
        enc={r['case_id']:r for r in encoded}
        encoding=dict(completed=True,passed=True,actual_checkpoint_CPU_processor=True,
            source_manifest_SHA256=source,Torch_CUDA_initialized=False,
            Native_constructed=False,weights_loaded=False,requests_file_SHA256=C.REQUEST_SHA,
            matching_body_pairs=64,cases=encoded)
        # These are explicitly recorded contract leaves, not actual CPU encode evidence.
        C.save(T/'encoding-recorded.json',encoding)
        ready=dict(ready=True,run=run,actual_step='CPU.actual-client',Native_SHA256=C.NATIVE,
            TP=8,PP=2,cards=16,rows=1,capacity=32768,
            loader_parent_source_manifest_SHA256=C.PARENT,service_profile_manifest_SHA256=profile,
            prepared_entry_identity=dict(upstream_commit=C.UPSTREAM),context_policy='v403-no-resident-V',
            context_policy_env='1',P0_text_buffered_until_completion=True,normal_output_budget=4096)
        C.save(T/'ready.json',ready)
        q=dict(actual_step=ready['actual_step'],run=run,Native_SHA256=C.NATIVE,
            source_manifest_SHA256=C.PARENT,service_profile_manifest_SHA256=profile,service_requests=[])
        C.save(owner,q)
        selected={r['case_id'] for i,r in enumerate(plan['requests']) if (i//2)%2==0}
        expected=[r for r in plan['comparison_schedule'] if r['case_id'] in selected]

        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                try:
                    assert self.path=='/v1/chat/completions'
                    raw=self.rfile.read(int(self.headers['Content-Length']))
                    r=bodies[C.digest(raw)];i=len(received);item=expected[i]
                    assert r['case_id']==item['case_id'] and raw==C.compact(r['body'])
                    assert self.headers['X-InfiniLM-Native-Profile']=='glm-nov16-w'+str(item['window'])
                    received.append((r['case_id'],item['window'],C.digest(raw)))
                    cap=i==2
                    ids=[21]*4096 if cap else [21,22,154820]
                    ticks=[10+j*.2 for j in range(len(ids))]
                    entry=enc[r['case_id']];response_id='CPU-response-'+str(i)
                    admission=dict(accepted_monotonic=9.,Native_prompt_token_ids_SHA256=entry['prompt_IDs_SHA256'],
                        history_render_audit=dict(qualified=True,prequeue_bound=True,markers_sent_to_Native=False,prompt_tokens=1))
                    row=dict(request_id=response_id,request_index=i,admitted_index=i,
                        native_profile='glm-nov16-w'+str(item['window']),prefill_control_window=item['window'],
                        service_profile_manifest_SHA256=profile,normal_reset_before=True,normal_reset_after=True,
                        original_reset_all16_completed=True,admission=admission,max_new_tokens=4096,
                        requested_output_budget=4096,prompt_tokens=1,actual_prompt_token_ids_SHA256=entry['prompt_IDs_SHA256'],
                        Native_continuation=[dict(kind='prefill',tokens=1)]+[dict(kind='decode',tokens=1)for _ in ids[1:]],
                        generated_token_ids=ids,actual_generated_tokens=len(ids),generated_token_ids_SHA256=C.digest(C.compact(ids)),
                        native_finish_reason='length'if cap else'eos_token',backend_token_monotonic=ticks,
                        started_monotonic=9.5,ended_monotonic=ticks[-1]+1,
                        Native_forward_host_summary=dict(prefill_calls=1,decode_calls=len(ids)-1),final_metadata=dict(recorded=True))
                    q['service_requests'].append(row);C.save(owner,q)
                    gold=cases[r['case_id']]['gold_numeric']
                    content='#### '+gold
                    if i==1:content='#### '+str(C.Fraction(gold)+1)
                    if i==3:content=None
                    reply=dict(id=response_id,choices=[dict(message=dict(role='assistant',content=content),
                        finish_reason='length'if cap else'stop')],usage=dict(prompt_tokens=1,completion_tokens=len(ids),total_tokens=1+len(ids)))
                    wire=C.compact(reply)
                    self.send_response(200);self.send_header('Content-Type','application/json')
                    self.send_header('Content-Length',str(len(wire)));self.end_headers();self.wfile.write(wire)
                except BaseException as error:
                    server_errors.append(repr(error));self.send_error(500)

        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever);thread.start()
        args=SimpleNamespace(base_url='http://127.0.0.1:'+str(server.server_port)+'/v1',
            ready=str(T/'ready.json'),service_report=str(owner),fixtures=str(F),
            encoding_receipt=str(T/'encoding-recorded.json'),encoding_sha=C.digest((T/'encoding-recorded.json').read_bytes()),
            encoding_source_sha=source,profile_manifest_sha=profile,loader_manifest_sha=C.PARENT,expected_step='CPU.actual-client',
            output_dir=str(T/'client-success'),shard_index=0,shards=2,native_offset=0,deadline_unix=time.time()+300,timeout=20)
        try:
            result=C.run(args)
            assert not server_errors,server_errors
            assert result['execution_passed'] and result['actual_HTTP']==len(received)==64
            assert result['summary']['complete_pairs']==32
            assert result['summary']['arms']['2']['correct']==31
            assert result['summary']['arms']['4']['correct']==30
            assert result['summary']['arms']['4']['caps']==1
            assert result['summary']['arms']['2']['parse_failures']==1
            assert result['summary']['paired_accuracy_W4_minus_W2']==-1/32
            assert result['summary']['W2_only_correct']==1 and result['summary']['W4_only_correct']==0
            assert result['summary']['noninferiority_proven'] is False
            assert result['rows'][0]['backend_visible_decode_tok_s']>4.99
            for which in ('encoding','step','fixture','loader'):
                bad=copy.copy(args);bad.output_dir=str(T/('rejected-'+which))
                if which=='encoding':bad.encoding_sha='0'*64
                elif which=='step':bad.expected_step='wrong-step'
                elif which=='loader':bad.loader_manifest_sha='0'*64
                else:bad.fixtures=str(T)
                before=len(received);failed=C.run(bad)
                assert not failed['execution_passed'] and failed['actual_HTTP']==0 and len(received)==before
        finally:
            server.shutdown();server.server_close();thread.join(timeout=5);assert not thread.is_alive()
        combined=[]
        for shard in (0,1):
            chosen={r['case_id']for i,r in enumerate(plan['requests'])if(i//2)%2==shard}
            rows=[r for r in plan['comparison_schedule']if r['case_id']in chosen]
            assert len(chosen)==32 and len(rows)==64
            assert sum(rows[i]['window']==2 for i in range(0,len(rows),2))==16
            assert sum(rows[i]['window']==4 for i in range(0,len(rows),2))==16
            combined.extend(chosen)
        assert len(combined)==len(set(combined))==64
        return dict(completed=True,passed=True,actual_HTTP_client_requests=64,
            same_body_header_pair_checks=32,whole_client_main_and_owner_file_handoff=True,
            quality_failures_retained=dict(wrong_answer=1,cap=1,parse_failure=1),
            preHTTP_negative_contracts=4,per_replica_balanced_pair_orders=True,
            two_shards_disjoint_complete64=True,
            recorded_leaves=['Native owner receipts','encoder contract','server model response'],
            real_checkpoint_encoded=False,actual_normal_FastAPI_executed=False,
            GPU=0,Native=0,Slurm=0,Model=0,
            client_SHA256=C.digest((D/'quality_client.py').read_bytes()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);assert not out.exists()
    result=check();out.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(dict(passed=True,output=str(out),SHA256=hashlib.sha256(out.read_bytes()).hexdigest())))
