"""New Kimi class/FIFO composition contracts, recording codec; not actual K3."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace


class Reasons:
    EOS_TOKEN='eos_token';LENGTH='length';TIMEOUT='timeout'
    CANCELED='canceled';ERROR='error'


class RecordingCodec:
    identity={'scope':'Recording only; not checkpoint tokenizer'}
    def encode_chat(self,messages,kwargs,generation):
        thinking=kwargs.get('thinking',kwargs.get('enable_thinking',True))
        text=json.dumps(messages,ensure_ascii=False,separators=(',',':'))
        if generation:text+='<|open|>'+('think'if thinking else'response')+'<|sep|>'
        return text,[ord(c)for c in text]
    def decode(self,ids):return ''.join(chr(i)for i in ids)


def engine_factory():
    from public_kimi_text.prepared import prepare
    return prepare('recording-model',Reasons,codec=RecordingCodec())['engine']


def admit(engine,mode='P0',ident='case'):
    data={'_agent_schemas':{},'_agent_choice':'none',
          'messages':[{'role':'user','content':'保持中文历史🙂'}],
          'chat_template_kwargs':{'enable_thinking':False},'max_tokens':4096}
    if mode is not None:data['text_delivery']=mode
    params=SimpleNamespace(temperature=0.,max_tokens=4096,stop=None)
    engine.validate_data(data,params)
    return engine.add_chat_request(messages=data['messages'],sampling_params=params,
        request_id=ident,request_data=data,chat_template_kwargs=data['chat_template_kwargs'])


async def stream_case(mode):
    engine=engine_factory();engine.start()
    seen=[];ready=threading.Event();release=threading.Event()
    def owner():
        req=engine.take(timeout=1)
        seen.append((req,req.metadata.get('text_delivery'),req.metadata['history_render_audit']['qualified']))
        ready.set()
        if not release.wait(2):raise RuntimeError('Recording owner timed out')
    thread=threading.Thread(target=owner);thread.start()
    req=admit(engine,mode)
    assert await asyncio.to_thread(ready.wait,1)
    release.set();thread.join(timeout=1)
    assert not thread.is_alive()and seen==[(req,mode or'P0',True)]
    stream=engine.stream_request(req,timeout=1,request_timeout=3)
    waiting=asyncio.create_task(anext(stream))
    req.emit_token(ord('A'))
    await asyncio.sleep(.02)
    first=None
    if mode=='P1':
        assert waiting.done();first=waiting.result()
        assert first.token_text=='A'and not first.finished and not req.is_finished()
    else:assert not waiting.done()
    req.emit_token(ord('中'));req.emit_token(163585)
    assert req.generated_token_ids==[ord('A'),ord('中'),163585]
    req.metadata['recording_owner_reset_completed']=True
    req.finish(Reasons.EOS_TOKEN)
    parts=[]if first is None else[first]
    if first is None:parts.append(await waiting)
    parts.extend([v async for v in stream])
    assert ''.join(v.token_text for v in parts)=='A中'and parts[-1].finished
    assert req.finish_reason==Reasons.EOS_TOKEN and not req.cancel_event.is_set()
    assert not engine._requests
    return {'case':'actual_Kimi_queue_composition','mode':mode or'defaultP0',
            'selection_before_owner_take':True,'early_text_before_finish':mode=='P1',
            'recording_owner_reset_before_done':True,'passed':True}


async def cancellation_case(kind):
    engine=engine_factory();req=admit(engine,'P1',kind);assert engine.take(timeout=1)is req
    stream=engine.stream_request(req,timeout=.01 if kind=='timeout'else 1,request_timeout=1)
    if kind=='timeout':
        result=await anext(stream);assert result.finish_reason==Reasons.TIMEOUT
    else:
        pending=asyncio.create_task(anext(stream));req.emit_token(ord('A'))
        assert (await pending).token_text=='A'
    await stream.aclose()
    assert req.cancel_event.is_set()and not req.is_finished()and not engine._requests
    req.metadata['recording_owner_reset_completed']=True;req.finish(Reasons.EOS_TOKEN)
    assert req.finish_reason==(Reasons.TIMEOUT if kind=='timeout'else Reasons.CANCELED)
    return {'case':kind,'abort_is_intent_not_owned_finish':True,'passed':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sys.path[:0]=[str(Path(__file__).resolve().parent),str(a.parent.resolve())]
    async def run():
        rows=[await stream_case(mode)for mode in [None,'P0','P1']]
        rows.extend([await cancellation_case(kind)for kind in ['timeout','disconnect']])
        return rows
    rows=asyncio.run(run())
    for value in ['P2',True,0,{},None]:
        engine=engine_factory()
        # Explicit null is invalid; omitted mode above is the default case.
        if value is None:
            from public_kimi_text.selector import selection
            try:selection({'text_delivery':None})
            except ValueError:pass
            else:raise AssertionError('Explicit null accepted')
        else:
            try:admit(engine,value)
            except ValueError:pass
            else:raise AssertionError('Invalid selection accepted')
        assert engine._pending.empty()and not engine._requests
    rows.append({'case':'invalid_delivery_before_FIFO','negative_values':5,'passed':True})
    result={'passed':True,'cases':rows,'scope':'Actual new prepared class + original Kimi queue/history with recording codec/owner, not actual tokenizer/parser/Native/HTTP/device','new_GPU':0}
    with a.output.open('x',encoding='utf-8')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'passed':True,'cases':len(rows),'scope':'recording contract only'}))


if __name__=='__main__':main()
