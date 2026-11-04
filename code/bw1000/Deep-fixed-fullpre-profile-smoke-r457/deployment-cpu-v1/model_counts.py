"""Actual fixed-profile per-request Native/86HC coverage, arbitrary prompt lengths."""
def validate(row,index):
 assert type(index)is int and index in[0,1]and row['r454_index']==index and row['r454_mode']==1 and row['r454_profile']=='deep-hc0-fullpre-c1'and row['r454_r366_fixed']==0
 assert row['r454_quiescent']is True and row['r454_poisoned']is False and row['normal_reset_before']and row['normal_reset_after']and not row.get('error')and not row.get('reset_error')
 f=row['actual_forward_completed'];assert type(f)is int and f>0 and f==row['actual_forward_attempts']
 prompt=row['prompt_tokens'];trace=row['Native_continuation'];prefill=[t for t in trace if t['kind']=='prefill'];decode=[t for t in trace if t['kind']=='decode_summary']
 assert [t['past']for t in prefill]==list(range(0,prompt,1024))and[t['tokens']for t in prefill]==[min(1024,prompt-p)for p in range(0,prompt,1024)]and all(t['total']==t['past']+t['tokens']for t in prefill)
 n=sum(t['calls']for t in decode);one=sum(t['tokens']==1 for t in prefill);assert len(decode)<=1 and f==len(prefill)+n and len(row['generated_token_ids'])==n+1
 if decode:assert decode[0]['first_past']==prompt and decode[0]['last_total']==prompt+n
 t1=86*(n+one);proof=row['r454_actual_T1_coverage'];assert proof==dict(passed=True,HC_groups=86,completed_forwards=f,one_token_prefill_forwards=one,decode_forwards=n,T1_HC_per_owner=t1,mode=1,r366_fixed=0)
 counts=row['r454_actual_owner_counts'];assert len(counts)==8
 for owner,c in enumerate(counts):
  assert c['owner']==owner and c['created']==c['ready']==86
  assert c['legacy']==dict(created=0,ready=0,pre=86*f,eligible_bf16_t1=t1,completed=0,other_t=86*(f-n-one),f32_t1=0)
  assert c['fullpre']==dict(observed=86*f,eligible=t1,selected=t1,completed=t1,prefill=86*(f-n-one),f32_t1=0,layout=0,selected_exceptions=0,fallback_exceptions=0,active=0)
 return proof
