"""Real private consumer,32 CPU owner threads,4-stage bounded mailboxes; no GPU."""
import hashlib
import json
from pathlib import Path
import queue
import threading
import weakref
from consumer import Row, decode_round, run


class Logits:
    def __init__(self, row, token, eos):
        self.row,self.token,self.eos=row,token,eos

    def argmax(self, axis):
        assert axis==-1
        # Distinct synthetic two-entry score row; never represents model logits.
        scores=[(self.token-1,-1.),(self.token,1.)]
        selected=max(scores,key=lambda v:v[1])[0]
        class Scalar:
            def item(self):return selected
        return Scalar()


def case(enabled, cancel=False):
    world=threading.Barrier(32,timeout=6)
    tp=[threading.Barrier(8,timeout=6) for _ in range(4)]
    mail=[[queue.Queue(1) for _ in range(8)] for _ in range(3)]
    ack=[[queue.Queue(1) for _ in range(8)] for _ in range(3)]
    results=[None]*32;errors=[]

    def owner(i):
        try:
            stage,rank=divmod(i,8);tid=threading.get_ident()
            rows=[Row(0,8912,101,8),Row(1,32760,201,8)]
            ledger={r.row:r.past for r in rows};steps={0:0,1:0};trace=[]
            live=weakref.WeakSet();peak=0;generated={0:[],1:[]}
            def forward(**kw):
                nonlocal peak
                assert threading.get_ident()==tid
                row,past,n=kw['descriptor']
                assert (kw['tokens'],kw['count'],n)==(1,1,1)
                assert kw['positions']==(past,) and kw['input_ids']==(rows[row].token,)
                assert past==ledger[row] and not rows[row].done
                key=(row,past,kw['input_ids']);trace.append(('forward',row,past));tp[stage].wait()
                if stage:
                    assert mail[stage-1][rank].get(timeout=6)==key
                    ack[stage-1][rank].put(key)
                if stage<3:
                    mail[stage][rank].put(key)
                    assert ack[stage][rank].get(timeout=6)==key
                ledger[row]+=1;steps[row]+=1
                logits=Logits(row,rows[row].token+1,steps[row]==(3 if row==0 else 5))
                live.add(logits);peak=max(peak,len(live))
                return logits,0.,0.,0.
            def feedback(row,logits):
                assert threading.get_ident()==tid and logits.row==row
                trace.append(('feedback',row));world.wait()
                selected=int(logits.argmax(-1).item())
                generated[row].append(selected);world.wait()
                return selected,logits.eos
            checks=[0];clean=[]
            def boundary():
                checks[0]+=1
                if cancel and checks[0]==2:raise InterruptedError('CPU agreed cancellation')
            def reset():
                world.wait();ledger.update({0:0,1:0});clean.append('reset');world.wait()
            def close():
                world.wait();clean.append('close');world.wait()
            try:run(rows,forward,feedback,reset,close,enabled=enabled,check=boundary,max_rounds=8)
            except InterruptedError:
                assert cancel
            else:assert not cancel and all(r.done for r in rows)
            assert ledger=={0:0,1:0} and clean==['reset','close']
            assert not list(live) and peak<=(2 if enabled else 1)
            results[i]={'trace':trace,'generated':generated,'past_after_reset':ledger,'peak_logits':peak,
                        'normal_owner_cleanup':True,'cancelled':cancel,'replay':False}
        except BaseException as error:
            errors.append((i,type(error).__name__,str(error)))
            world.abort()
            for b in tp:b.abort()
    threads=[threading.Thread(target=owner,args=(i,)) for i in range(32)]
    for t in threads:t.start()
    for t in threads:t.join(10)
    assert not any(t.is_alive() for t in threads) and not errors,errors
    assert all(v['trace']==results[0]['trace'] and v['generated']==results[0]['generated'] for v in results)
    return results[0]


def main():
    off,on=case(False),case(True)
    assert off['generated']==on['generated']
    assert [x[0] for x in off['trace'][:4]]==['forward','feedback','forward','feedback']
    assert [x[0] for x in on['trace'][:4]]==['forward','forward','feedback','feedback']
    bad=0
    for rows in [[Row(0,1,1,2),Row(0,2,2,2)], [Row(0,32768,1,2),Row(1,2,2,2)],
                 [Row(0,1,-1,2),Row(1,2,2,2)], [Row(0,1,1,0),Row(1,2,2,2)]]:
        calls=[]
        try:decode_round(rows,lambda **k:calls.append(k),lambda *a:None,enabled=True)
        except ValueError:bad+=1
        assert not calls
    failed=[]
    def fail_forward(**kw):raise RuntimeError('Mock Native failure; never resume partial round')
    try:run([Row(0,1,1,2),Row(1,2,2,2)],fail_forward,lambda *a:None,
            lambda:failed.append('reset'),lambda:failed.append('close'),enabled=True)
    except RuntimeError:pass
    assert failed==['reset','close']
    canceled=case(True,True)
    from check_binding_CPU import check_binding
    binding=check_binding()
    q={'completed':True,'passed':True,'GPU':0,'Native_dlopen':False,'owner_threads':32,
       'PP':4,'TP':8,'real_consumer_executed':True,'mock_Native_CCL_and_logits':True,
       'Native_numeric_or_memory_qualification':False,'Model_gain':'UNKNOWN',
       'OFF':off,'ON':on,'cancelled_pair':canceled,'actual_binding_CPU':binding,'invalid_precallback_cases':bad,
       'Native_failure_cleanup_no_replay':failed,
       'source_SHA256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    out=Path(__file__).with_name('CPU-result.json');out.write_text(json.dumps(q,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'passed':True,'GPU':0,'owners':32,'invalid_precallback_cases':bad}))


if __name__=='__main__':main()
