"""One finite full-logit reference -> token-only replay, already-loaded Engine.

Existing outer runner must first pass original B16M4 historical anchor and8QA.
Uses existing reset/barriers, no new broker/scheduler, no Model/SDK compilation.
This function is CPU-prepared only; it has not executed on GPU.
"""
import hashlib,json,os,pathlib,time
from kimi_matched_trace_r8467 import digest,seal_reference,read_reference,replay
from online_kimi_provider_r8467 import KimiLoadedAdapter


class BorrowedProvider(KimiLoadedAdapter):
    def close(self):self.closed=True # outer old runner owns actualnative destroy.


def run_matched_static(g,inputs,eos,reset):
    assert len(inputs)==8 and (g['tp_nodes'],g['pp'])==(1,4)
    lib,ctx,r,node,tag=g['lib'],g['ctx'],g['r'],g['node'],g['tag']
    report,control,save=g['report'],g['control'],g['save']
    assert report['historical_B16M4_anchor_passed'] and report['before8QA_passed']
    assert os.environ['INFINILM_KIMI_MOE_EXECUTION_R4665']=='1'
    flags={k:os.environ.get(k) for k in ['INFINILM_KIMI_N4_TILE_R8167','INFINILM_KIMI_VOCAB_HEAD_R8341','INFINILM_KIMI_PACKED_AUTO_R6136','INFINILM_KIMI_METADATA_PACK_R5858','INFINILM_KIMI_BOUNDED_BF16_R6054','INFINILM_KIMI_STATIC_DIV_R6020']}
    assert all(v=='0' for v in flags.values())
    identity=dict(DSO_SHA256=hashlib.sha256(g['library'].read_bytes()).hexdigest(),
                  SDK_SHA256=report['libinfiniops.so_sha256'],TP=8,PP=4,rows=32,pages_per_request=4,
                  config_SHA256=hashlib.sha256((g['model']/'config.json').read_bytes()).hexdigest(),
                  flags=flags,serving_policy=1,inputs_SHA256=digest(inputs))
    header=dict(scope='MATCHED_SHAPE_JOBTRACE_ONLY',identity=identity,rows=32,reference_flags=67,
                candidate_flags=71,eos_token_ids=sorted(eos),output_limits={str(k):128 for k in range(8)})
    def memory_gate():assert all(free>512*1024**2 for free,total in g['memory']())
    def provider(full):return BorrowedProvider(lib,ctx,stage=node,stages=4,rows=32,check=g['check'],memory_gate=memory_gate,destroy=lambda p:None,full_logit_reference=full)
    reset('matched-full-logit-reference')
    reference=provider(True);records=[];outputs=[[] for _ in inputs];pasts=[0]*8
    start=time.perf_counter();report['phase']='matched-full-logit-reference';save()
    trace_log=r/'results'/f'Kimi-matched-r8468-{tag}-node{node}-actuals.jsonl'
    with trace_log.open('x')as journal:
        def call(ii,ins):
            job=dict(seq=len(records),kind='prefill' if all(pasts[i]==0 for i in ii) else 'decode',
                     keys=ii,rows=[i+1 for i in ii],pasts=[pasts[i] for i in ii],inputs=ins)
            tick=time.perf_counter();native=reference.forward(job)
            values=control.collect('matched-reference-'+str(job['seq']),dict(job_SHA256=digest(job),ids=native,oracle=reference.reference_observation if node==3 else None))
            assert all(v['job_SHA256']==digest(job) for v in values)
            tail=values[3];oracle=tail['oracle']
            record=dict(job=job,job_SHA256=digest(job),reference_ids=tail['ids'],oracle=oracle,
                        phase_elapsed_s=time.perf_counter()-start,call_and_collect_s=time.perf_counter()-tick)
            # Persist fullactuals/top2/margins BEFORE finite/CPUargmax assertion.
            journal.write(json.dumps(record)+'\n');journal.flush();os.fsync(journal.fileno())
            records.append(record);report['matched_reference_active']=dict(records=len(records),last=record,elapsed_s=time.perf_counter()-start);save()
            assert oracle['finite'] and oracle['native_equals_CPU'],'full-logit native versus deterministic CPU argmax mismatch'
            for i,ins_row,token in zip(ii,ins,tail['ids']):outputs[i].append(token);pasts[i]+=len(ins_row)
        for a in range(0,8,2):
            ii=list(range(a,a+2));call(ii,[inputs[i] for i in ii])
        for _ in range(127):
            active=[i for i in range(8) if len(outputs[i])<128 and outputs[i][-1]not in eos]
            if not active:break
            for a in range(0,len(active),4):
                ii=active[a:a+4];call(ii,[[outputs[i][-1]] for i in ii])
    reference.close()
    report['matched_reference_complete_actuals']=outputs
    report['matched_reference_phase_s']=time.perf_counter()-start;save()
    path=r/'results'/f'Kimi-matched-r8468-{tag}-reference.json'
    h=seal_reference(path,header,records) if node==3 else None
    packet=control.collect('matched-reference-sealed',dict(SHA256=h))
    h=packet[3]['SHA256'];sealed=read_reference(path,h,identity)
    reset('matched-token-only-replay')
    candidate=provider(False);start=time.perf_counter();report['phase']='matched-token-only-replay';save()
    def persist(value):
        report['matched_replay']=value;report['matched_replay_phase_s']=time.perf_counter()-start;save()
    try:
        result=replay(sealed,candidate,control.collect,persist)
    finally:candidate.close()
    report.update(matched_trace_result=result,matched_trace_SHA256=h,matched_trace_source=str(path),
                  matched_identity=identity,
                  matched_trace_scope='Numerical MATCHED_SHAPE_JOBTRACE_ONLY. SameDSO full-logit CPUargmax reference; not independent modelgold, not live online performance; no arbitrarygeometry claim.')
    save()
    return result
