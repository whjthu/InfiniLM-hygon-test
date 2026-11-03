"""Real original bridge/request consumer, with only byte tokenizer as a leaf."""
from pathlib import Path
import argparse, asyncio, hashlib, importlib.util, json
from types import SimpleNamespace


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


async def check(repo):
    here=Path(__file__).parent
    original=repo/'code/bw1000/public-GLM-native-provider-r425/public_glm_native/native_engine_bridge_r268.py'
    base=module('original_native_queue_r429_CPU',original)
    candidate=module('incremental_text_r429_CPU',here/'incremental_text.py')
    reasons=SimpleNamespace(TIMEOUT='timeout',ERROR='error',CANCELED='canceled',EOS_TOKEN='eos')
    class ByteTokenizer:
        def decode(self, ids, **kwargs):return bytes(ids).decode('utf8',errors='replace')
    class Engine(candidate.IncrementalTextMixin,base.NativeEngineBridge):pass
    def request(decoder=None):
        e=Engine(ByteTokenizer(),32768,reasons,decode_tokens=decoder,eos_token_ids=(999,),default_max_tokens=128)
        r=base.NativeRequest('original-request','prompt',[1],128,(999,),reasons)
        e._requests[r.request_id]=r
        return e,r
    cases=[]
    # An actual first progress delta is delivered while NativeRequest remains
    # unfinished. EOS and final transport are emitted only after owner finish.
    e,r=request();r.emit_token(65)
    stream=e.stream_request(r,timeout=1,request_timeout=2)
    first=await anext(stream);assert first.token_text=='A'and not first.finished and not r.is_finished()
    r.emit_token(66);second=await anext(stream);assert second.token_text=='B'and not second.finished
    r.emit_token(999);r.metadata['recording_native_reset_completed']=True;r.finish(reasons.EOS_TOKEN)
    rest=[v async for v in stream]
    assert ''.join([first.token_text,second.token_text]+[v.token_text for v in rest])=='AB'
    assert rest[-1].finished and rest[-1].finish_reason==reasons.EOS_TOKEN
    assert r.generated_token_ids==[65,66,999]and r.visible_token_ids==[65,66]and not r.cancel_event.is_set()
    assert r.request_id not in e._requests
    cases.append('real_NativeRequest_first_delta_before_finish_and_EOS_after_owned_reset')
    # Three BPE byte pieces complete one Chinese character; no replacement tail
    # is leaked to the transport while the prefix is incomplete.
    e,r=request();stream=e.stream_request(r,timeout=1,request_timeout=2)
    pending=asyncio.create_task(anext(stream))
    for v in '中'.encode():r.emit_token(v);await asyncio.sleep(.005)
    first=await pending;assert first.token_text=='中'and not first.finished
    r.finish(reasons.EOS_TOKEN);rest=[v async for v in stream]
    assert ''.join([first.token_text]+[v.token_text for v in rest])=='中'
    cases.append('split_UTF8_prefix_delivered_without_replacement_character')
    # Context-sensitive prefix rewrites fail rather than corrupt an already
    # emitted HTTP body; transport abort is intent, not a forged Native finish.
    e,r=request(lambda ids:'A'if len(ids)==1 else'B');r.emit_token(65)
    stream=e.stream_request(r,timeout=1,request_timeout=2);assert(await anext(stream)).token_text=='A'
    r.emit_token(66)
    try:await anext(stream)
    except RuntimeError as error:assert 'rewrote' in str(error)
    else:raise AssertionError('prefix rewrite accepted')
    assert r.cancel_event.is_set()and r.abort_reason==reasons.ERROR and not r.is_finished()
    cases.append('decoder_rewrite_rejected_with_ERROR_intent_without_finish')
    # Explicit async generator close represents disconnect. Original owner
    # still owns reset and finish, so it can complete them after cancellation.
    e,r=request();r.emit_token(65);stream=e.stream_request(r,timeout=1,request_timeout=2)
    await anext(stream);await stream.aclose()
    assert r.cancel_event.is_set()and r.abort_reason==reasons.CANCELED and not r.is_finished()
    r.metadata['recording_native_reset_completed']=True;r.finish(reasons.EOS_TOKEN)
    assert r.finish_reason==reasons.CANCELED
    cases.append('disconnect_requests_cancel_but_owner_retains_reset_finish')
    # Artificially advancing only this module's clock does not alter asyncio's
    # real clock. A busy original progress queue cannot postpone the deadline.
    e,r=request();[r.emit_token(65)for _ in range(30)]
    ticks=iter(i*.01 for i in range(1000));candidate._monotonic=lambda:next(ticks)
    out=[v async for v in e.stream_request(r,timeout=1,request_timeout=.11)]
    assert out[-1].finish_reason==reasons.TIMEOUT and r.metadata['transport_timeout']
    assert r.abort_reason==reasons.TIMEOUT and not r.is_finished()
    candidate._monotonic=__import__('time').monotonic
    cases.append('busy_progress_does_not_slide_absolute_deadline')
    e,r=request();r.emit_token(65);stream=e.stream_request(r,timeout=1,request_timeout=2)
    await anext(stream);r.fail(RuntimeError('recording_native_failure'))
    try:await anext(stream)
    except RuntimeError as error:assert 'recording_native_failure'in str(error)
    else:raise AssertionError('worker error hidden')
    assert r.finish_reason==reasons.ERROR
    cases.append('real_worker_error_event_not_converted_to_normal_EOS')
    # Genuine U+FFFD can be a real model output and must survive final flush.
    prefix=candidate.TextPrefix(lambda ids:'\ufffd')
    assert prefix.advance([1],final=False)==''and prefix.advance([1],final=True)=='\ufffd'
    cases.append('literal_replacement_character_preserved_on_final_flush')
    # Coalesced progress is one delta, not a fictitious sequence of timestamps.
    e,r=request();[r.emit_token(x)for x in [65,66,67]];r.finish(reasons.EOS_TOKEN)
    out=[v async for v in e.stream_request(r,timeout=1,request_timeout=2)]
    assert ''.join(v.token_text for v in out)=='ABC'and len(out)==2
    assert r.metadata['text_publish_records'][0]['cumulative_visible_token_ids']==3
    cases.append('coalesced_progress_preserves_source_count_not_fake_token_clock')
    return dict(passed=True,cases=cases,actual_bridge_SHA256=hashlib.sha256(original.read_bytes()).hexdigest(),
        actual_consumer='Explicit mixin + unchanged production NativeEngineBridge and NativeRequest',
        recording_leaves=['ByteTokenizer only; no actual checkpoint tokenizer','Native owner actions simulated by original emit/finish/fail methods'],
        new_jobs=0,GPU=False,Native_dlopen=False,Model_READY=False,
        actual_public_HTTP_SSE_qualification=False,actual_checkpoint_tokenizer_qualification=False,
        baseline_providers_modified=False,performance_claim=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists();q=asyncio.run(check(a.repo))
    a.out.write_text(json.dumps(q,indent=2)+'\n',encoding='utf8')
    print(json.dumps(q))
