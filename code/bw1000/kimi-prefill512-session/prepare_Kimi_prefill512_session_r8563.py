from pathlib import Path
import ast
def write(name,s):
 Path(name).write_text(s,encoding='utf-8',newline='\n')
 if name.endswith('.py'):ast.parse(s)

# Reuse the tested finite TCP owner loop, timing/export and lifecycle. New
# arrivals use preattached independent ingress connections so an earlier ACK
# cannot hold back release of another request. Actual ingress remains recorded.
s=Path('kimi_bounded_batch_wait_r8485.py').read_text().replace('r8485','r8563')
s=s.replace('from kimi_matched_static_r8468 import BorrowedProvider','from kimi_prefill512_scope_r8563 import ScopedProvider as BorrowedProvider')
a=s.index('def emit_fixed_arrivals(');b=s.index('\ndef bounded_epoch',a)
s=s[:a]+'''def emit_fixed_arrivals(c,inputs,arrival_plan,records,*,clock=time.perf_counter,sleep=time.sleep):
    """Independent actual ingress streams; no previous-request ACK dependency."""
    address,nonce=c
    clients=[Client(address,nonce,'ingress')for _ in inputs]
    base=clock();errors=[];lock=threading.Lock()
    def one(i):
        target=base+arrival_plan['offsets_s'][i]
        try:
            assert digest(inputs[i])==arrival_plan['prompt_SHA256'][i]
            while True:
                remain=target-clock()
                if remain<=0:break
                sleep(min(remain,.001))
            sent=clock();got=clients[i].call(dict(op='arrive',prompt=inputs[i],limit=128))
            with lock:records.append(dict(producer_index=i,request=got['request'],target_t=target,send_t=sent,lateness_s=sent-target,actual_ingress_t=got['arrival']))
            assert got['accepted']
        except BaseException as e:
            with lock:errors.append(repr(e))
        finally:clients[i].close()
    threads=[threading.Thread(target=one,args=(i,),daemon=True)for i in range(len(inputs))]
    for t in threads:t.start()
    for t in threads:t.join(20)
    assert not any(t.is_alive()for t in threads)and not errors,errors
    assert sorted(v['request']for v in records)==list(range(len(inputs)))
    records.sort(key=lambda v:v['producer_index'])

''' +s[b:]
s=s.replace("emit_fixed_arrivals(c,inputs,arrival_plan,producer_schedule)","emit_fixed_arrivals((address,nonce),inputs,arrival_plan,producer_schedule)")
s=s[:s.index('\ndef consume_plan(')]
s=s.replace("shared=dict(captured=captured,outputs=outputs,event_path=str(event_path),capture_path=str(capture_path))","shared=dict(captured=captured,outputs=outputs,producer_schedule=producer_schedule,metric=metric,event_path=str(event_path),capture_path=str(capture_path))")
write('kimi_prefill512_live_r8563.py',s)

s=Path('modern_Kimi_missing_detail_r8533.py').read_text().replace('missing-detail-r8533','prefill512-r8563').replace('Kimi-missing-detail-manifest-r8533.json','Kimi-prefill512-manifest-r8563.json').replace('kimi_missing_detail_protocol_r8533','kimi_prefill512_protocol_r8563').replace('Kimi_native_binding_r8521','Kimi_prefill512_binding_r8563')
s=s.replace("report['numerical_variant']", "report['numerical_variant']")
write('modern_Kimi_prefill512_r8563.py',s)
s=Path('Kimi_missing_detail_r8533_inside.sh').read_text().replace('modern_Kimi_missing_detail_r8533.py','modern_Kimi_prefill512_r8563.py')
s=s.replace('python /refactor/incoming/Kimi_Q0_idle_r8066.py','export INFINILM_KIMI_PREFILL512_R8562=0 INFINILM_KIMI_PREFILL512_ELIGIBLE_R8562=0\npython /refactor/incoming/Kimi_Q0_idle_r8066.py')
s=s.replace('2000 python','3500 python')
write('Kimi_prefill512_r8563_inside.sh',s)
