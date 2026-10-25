import datetime,hashlib,json,pathlib,statistics
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
name='Kimi-interleaved-service-r8669aaa'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def dist(v):return dict(min=min(v),mean=statistics.mean(v),max=max(v)) if v else None
wp=r/(name+'-watch.json');w=read(wp);run=w['run']
ap=r/'results'/f'{name}-audit-{run}.json';audit=read(ap)
paths=[r/'results'/f'kimi-interleaved-service-r8669aaa-{run}-node{n}.json' for n in range(4)]
reports=[read(p) for p in paths];emitter=reports[2]
a=next(z for z in emitter['serving_cases'] if z['label']=='TP16-B16M16-A-current-ordinary128')
cs=emitter['interleaved_cases'];assert len(cs)==1
c=cs[0]
def summary(z):
    out={k:z.get(k) for k in ['label','B','frontend_cohort_M','aggregate_decode_tok_s','decode_tok_s_per_card','prefill_wall_seconds','decode_wall_seconds','service_wall_seconds','end_to_end_output_tok_s','end_to_end_output_tok_s_per_card','matched_independent_complete_IDs_EOS','timing_scope','user_clock']}
    out['per_user_first_last_tok_s']=dist(z['per_request_first_last_tok_s'])
    out['TTFT_seconds']=dist(z.get('TTFT_seconds',[]))
    out['max_ITL_seconds']=dist(z['per_request_max_ITL_seconds'])
    return out
out=dict(hardware='BW1000',level='WHOLE_MODEL_FINITE_INTERLEAVED_SCHEDULE',
 status='COMPLETED_CANDIDATE_FINAL_TOKEN_NEGATIVE',actual_handle=w['actual_handle'],run=run,
 observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit='d645f7e2c6b95079b4e5d91af5f92057909b9703',
 native_SHA256=emitter['library_sha256'],terminal=w['terminal'],original_audit_exit=w['audit_exit'],
 original_audit_error=audit.get('error'),original_audit_qualified=audit['qualified'],
 baseline_and_lifecycle_qualified=audit.get('baseline_and_lifecycle_qualified',False),
 all32normal=audit['all32normal'],QA_counts=[len(q['QA']) for q in reports],
 QA_all_frozen_tokens_exact=all(z['complete_frozen_IDs_exact'] for q in reports for z in q['QA']),
 post_KFD_passed=w['post_KFD_passed'],unique_post=w['post_KFD'],
 matched_frozen648_original_A=all(q['frozen648_original_reference']['matched_full_IDs_EOS'] for q in reports),
 candidate_final_tokens_qualified=False,performance_adopted=False,
 A_ordinary=summary(a),C_mixed_diagnostic_only=summary(c),
 C_full128_rows=len(c['generated_ids'])==16 and all(len(v)==128 for v in c['generated_ids']),
 C_first_differences=c['first_differences'],C_mismatched_requests=len(c['first_differences']),
 C_repeat=emitter.get('interleaved_repeat_status'),
 loading_seconds=[q.get('loading_seconds') for q in reports],
 network=audit.get('network_logs'),
 numerical_cause='UNRESOLVED: mixed geometry uses generic earlyN2..14 then acceptedI192N16. No layer-level runtime comparison proves which path caused later token changes.',
 decision='Reject this candidate. Keep accepted ordinaryA. No repeat/load/profile automatically authorized.',
 timing_scope='Same emitter perf_counter first actual prefill token through last useful token, no cross-host subtraction/HTTP claim. C earlydecode overlaps laterprefill; C puredecode TPS is not compared to A aggregate_decode TPS.',
 workload='TP16PP2/32DCU cap16/2 natural512 prompts repeated16/full128; not16distinct traffic or a broad serving frontier.',
 evidence=dict(watch_SHA256=sha(wp),audit_SHA256=sha(ap),report_SHA256=[sha(p) for p in paths]))
assert out['all32normal'] and out['QA_counts']==[16]*4 and out['post_KFD_passed']
assert out['baseline_and_lifecycle_qualified'] and out['matched_frozen648_original_A'] and out['C_full128_rows']
assert not audit['qualified'] and c['first_differences']
print(json.dumps(out))
