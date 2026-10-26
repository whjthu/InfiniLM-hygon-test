import json, pathlib, hashlib, collections, statistics, datetime

ROOT=pathlib.Path(r'C:\Users\WangHaojie\Documents\ChatGPT\海光集群')
OUT=pathlib.Path(__file__).resolve().parent
BASE=ROOT/'GLM-down-row2-Model-private-r9284b'
def read(p): return json.loads(p.read_bytes())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dist(v):
    v=sorted(v)
    def q(f):
        x=(len(v)-1)*f; a=int(x); b=min(a+1,len(v)-1)
        return v[a]+(v[b]-v[a])*(x-a)
    return dict(n=len(v), min=v[0], mean=statistics.mean(v), p50=q(.5), p95=q(.95), p99=q(.99), max=v[-1]) if v else None

canonical=read(BASE/'terminal-canonical-r9287.json')
assert sha(BASE/'terminal-canonical-r9287.json')=='540d2616207eac39b5c7e29c6405d71ae304b65698963bfb22a9d9a745f8fd05'
review=OUT.parent/'astra-review-115-current.json'
assert sha(review)=='5b5a91b011d3c28e29dca102a1e0344501bd5aed2e7ed8b70368eb8a94a2636c'
evidence=read(BASE/'terminal-evidence-r9287.json')
projection=read(BASE/'arrival-terminal-projection-r9286.json')
arms=[]
for a,p in zip(evidence['audit']['episodes'],projection['arms']):
    assert a['label']==p['label']
    calls=a['native_calls']; groups=p['native_phases']; intervals={k:iter(v['intervals_from_first_any_CAPI']) for k,v in groups.items()}
    for c in calls:
        lo,hi=next(intervals[c['op']]); c['_seconds']=hi-lo; c['_T']=sum(c['counts']); c['_B']=len(c['rows'])
    sums={k:sum(c['_seconds'] for c in calls if c['op']==k) for k in groups}
    assert all(abs(sums[k]-groups[k]['HOST_CAPI_wall_sum_seconds'])<1e-7 for k in groups)
    shape=collections.defaultdict(lambda:dict(calls=0,HOST_CAPI_seconds=0.,input_tokens=0))
    for c in calls:
        key=f"{c['op']}/T{c['_T']}/B{c['_B']}"; v=shape[key]; v['calls']+=1; v['HOST_CAPI_seconds']+=c['_seconds']; v['input_tokens']+=c['_T']
    waits=[]
    for r in a['serving']['requests']:
        row=r['row']; events=r['events']; n=len(events)
        anchor=next(c for c in calls if c['op']=='ANCHOR' and row in c['rows'])
        dec=[c for c in calls if c['op']=='DECODE' and row in c['rows']]
        assert len(dec)==len(r['generated_ids'])-1
        # No token values are inspected or copied. EOS is the final generated id;
        # its absence from visible events gives the last visible command index.
        assert r['stop_reason']=='EOS' and n>=2
        last=dec[n-2]
        covered=[c for c in calls if anchor['seq']<c['seq']<=last['seq']]
        budgets={k:sum(c['_seconds'] for c in covered if c['op']==k) for k in groups}
        wall=events[-1]['time']-events[0]['time']
        assert wall+1e-7>=sum(budgets.values())
        assert all(row in c['rows'] for c in covered if c['op']=='DECODE')
        waits.append(dict(row=row,arrival=r['arrival'],visible_tokens=n,firstlast_seconds=wall,firstlast_tok_s=(n-1)/wall,
            admission_queue_seconds=r['admitted']-r['arrival'],admission_to_dispatch_seconds=r['service_start']-r['admitted'],
            dispatch_to_first_seconds=events[0]['time']-r['service_start'],
            contained_HOST_CAPI_seconds=budgets,outside_HOST_CAPI_seconds=wall-sum(budgets.values()),
            first_anchor_seq=anchor['seq'],last_visible_seq=last['seq']))
    B=collections.defaultdict(lambda:dict(calls=0,seconds=0.,tokens=0))
    for c in calls:
        if c['op']=='DECODE':
            v=B[c['_B']];v['calls']+=1;v['seconds']+=c['_seconds'];v['tokens']+=c['_T']
    wall=p['service_wall_seconds']
    arms.append(dict(label=a['label'],row2_ON=a['down_row2'],service_wall_seconds=wall,phase_HOST_CAPI_seconds=sums,
        outside_HOST_CAPI_seconds=wall-sum(sums.values()),phase_fraction_of_service_wall={k:v/wall for k,v in sums.items()},
        decode_actual_T_and_B=B,actual_shapes=shape,
        row2_eligible_decode=dict(calls=sum(c['op']=='DECODE' and c['_T'] in (8,16) for c in calls),
            HOST_CAPI_seconds=sum(c['_seconds'] for c in calls if c['op']=='DECODE' and c['_T'] in (8,16)),
            tokens=sum(c['_T'] for c in calls if c['op']=='DECODE' and c['_T'] in (8,16))),
        requests=waits,request_distributions={k:dist([r[k] for r in waits]) for k in ['firstlast_seconds','firstlast_tok_s','admission_queue_seconds','admission_to_dispatch_seconds','dispatch_to_first_seconds','outside_HOST_CAPI_seconds']},
        request_contained_CAPI_distributions={k:dist([r['contained_HOST_CAPI_seconds'][k] for r in waits]) for k in groups}))

pooled={}
for on in (False,True):
    aa=[a for a in arms if a['row2_ON']==on]; wall=sum(a['service_wall_seconds'] for a in aa)
    sums={k:sum(a['phase_HOST_CAPI_seconds'][k] for a in aa) for k in ['PREFIX','ANCHOR','DECODE']}
    pooled['ON' if on else 'OFF']=dict(service_wall_seconds=wall,phase_HOST_CAPI_seconds=sums,
        outside_HOST_CAPI_seconds=wall-sum(sums.values()),phase_fraction_of_service_wall={k:v/wall for k,v in sums.items()},
        request_contained_CAPI_distributions={k:dist([r['contained_HOST_CAPI_seconds'][k] for a in aa for r in a['requests']]) for k in sums},
        request_firstlast_seconds=sum(r['firstlast_seconds'] for a in aa for r in a['requests']),
        request_firstlast_prefill_containment_seconds=sum(r['contained_HOST_CAPI_seconds']['PREFIX']+r['contained_HOST_CAPI_seconds']['ANCHOR'] for a in aa for r in a['requests']))

sources=[review,BASE/'terminal-canonical-r9287.json',BASE/'terminal-evidence-r9287.json',BASE/'arrival-terminal-projection-r9286.json',
    BASE/'glm_arrival_probe_r9284b.py',BASE/'arrival_runner_r9258.py',BASE/'source-pins.json',BASE/'GPU-manifest.json',BASE/'audit.py',
    ROOT/'GLM-down-row2-Model-private-r9284/native_CPU_build.py',ROOT/'GLM-down-row2-Model-private-r9284/dispatch-row2-Model-r9284.cpp',
    ROOT/'GLM-B32-Model-private-r9183/CPU-build.json',ROOT/'GLM-owner-profile-hooks-r9172.json',
    ROOT/'glm-window-r5797-src/glm_moe_dsa_balanced_text_model_r126.cpp',ROOT/'glm_model_window_r5797.cpp',
    ROOT/'GLM-decode-profile-private-r8812/glm_decode_window_observer_r8812.cpp',
    ROOT/'InfiniLM-hygon-test/results/bw1000/2026-10-26/glm719-deferred-feedback-qualified.json',
    ROOT/'InfiniLM-hygon-test/results/bw1000/2026-10-26/glm724-prefill-pipeline-qualified.json',
    ROOT/'InfiniLM-hygon-test/results/bw1000/2026-10-25/glm582-prepared-weight-reuse-component.json']
index=dict(schema=1,files=[dict(path=str(p),SHA256=sha(p),bytes=p.stat().st_size) for p in sources])
(OUT/'source-hash-index-r118.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
result=dict(schema=1,action='G115A',owner='glm_residual_closure_118',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    actual_handle='910376.755',actual_native_SHA256=canonical['Model_SHA256'],scope=canonical['scope'],
    source_hash_index_SHA256=sha(OUT/'source-hash-index-r118.json'),arms=arms,pooled=pooled,
    measurement_caveats=['Only emitter stage1/rank0 per-call phase-duration projection is local; all16 matching command traces are qualified, but all16 timing arrays are absent locally.',
        'Request containment partitions existing first-last wall by completed CAPI commands between first anchor and last visible decode. PREFIX/ANCHOR time is serial blocking inside these request timelines, not extra time to add to service wall.',
        'HOST CAPI includes PP/TP waits, communication, submission and original drains. Outside CAPI includes frontend/control/statistics/allocation/emission overhead; it is not pure CPU or network.',
        'Three existing HOST pipeline phases are nested inside CAPI. Their shifts cannot prove physical compute, GPU busy, or PP latency; incoming recv enqueue and later layers/sync may absorb dependency waits.',
        'Physical bandwidth peak and current native-math/TP/PP exclusive shares are UNKNOWN. Old540 M8 and743 synthetic T16 do not fill them.'])
(OUT/'residual-budget-r118.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(pooled=pooled,arm_summary=[{k:a[k] for k in ['label','service_wall_seconds','phase_HOST_CAPI_seconds','outside_HOST_CAPI_seconds','row2_eligible_decode','request_contained_CAPI_distributions']} for a in arms]),ensure_ascii=False,indent=2))
