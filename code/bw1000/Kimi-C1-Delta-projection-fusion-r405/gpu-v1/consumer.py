"""Shared actual full-Delta call schedule. Backend is real CABI on GPU, recording on CPU."""

def schedule():
    return [(r,on,t) for r in range(8) for on in ([False,True,True,False] if r%2==0 else [True,False,False,True]) for t in range(10)]

def execute(backend,save,cancelled=lambda:False):
    records=[]
    for index,(round_id,on,trial) in enumerate(schedule()):
        if cancelled():raise RuntimeError('cancelled before next full Delta call')
        backend.restore(); backend.mode(on); backend.barrier(index)
        before=backend.stats()
        raw=backend.run(trial%2)
        after=backend.stats()
        diag=backend.inspect(raw,trial%2)
        counts_ok=(after['chosen']-before['chosen']==int(on) and after['fallback']==before['fallback'] and all(b-a==int(on) for a,b in zip(before['consumer_counts'],after['consumer_counts'])))
        row=dict(index=index,round=round_id,arm='ON' if on else 'OFF',trial=trial,warmup=trial<2,call_completed=True,counters_before=before,counters_after=after,numerical=diag,passed=bool(counts_ok and diag['passed']),counter_passed=counts_ok)
        records.append(row); save(row)
        if not row['passed']:raise RuntimeError('completed full Delta call failed qualification; saved before STOP')
    return records
