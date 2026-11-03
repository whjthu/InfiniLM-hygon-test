"""Independent passive86HC/8owner proof at actual request completion/drain."""
MODES=(0,1,0,1,1,0)

def validate(row,index):
 assert type(index)is int and 0<=index<6
 mode=MODES[index];assert row['r449_index']==index and row['r449_mode']==mode and row['r449_warm']==(index<2)and row['r449_r366_fixed']==0
 assert row['r449_quiescent']is True and row['r449_poisoned']is False
 assert row['normal_reset_before']and row['normal_reset_after']and not row.get('error')and not row.get('reset_error')
 f=row['actual_forward_completed'];assert type(f)is int and f>0 and f==row['actual_forward_attempts']
 trace=row['Native_continuation'];prefill=[x for x in trace if x['kind']=='prefill'];decode=[x for x in trace if x['kind']=='decode_summary']
 assert len(prefill)==9 and [x['past']for x in prefill]==list(range(0,8489,1024))and[x['tokens']for x in prefill]==[1024]*8+[297]
 assert [x['total']for x in prefill]==[x['past']+x['tokens']for x in prefill]
 n=sum(x['calls']for x in decode);assert len(decode)<=1 and f==9+n and len(row['generated_token_ids'])==n+1
 if decode:assert decode[0]['first_past']==8489 and decode[0]['last_total']==8489+n
 t1=86*n;proof=row['r449_actual_T1_coverage']
 assert proof==dict(passed=True,HC_groups=86,completed_forwards=f,one_token_prefill_forwards=0,decode_forwards=n,T1_HC_per_owner=t1,mode=mode,r366_fixed=0)and t1>0
 counts=row['r449_actual_owner_counts'];assert len(counts)==8
 for owner,c in enumerate(counts):
  assert c['owner']==owner and c['created']==c['ready']==86
  a,b=c['legacy'],c['fullpre'];assert all(type(v)is int and v>=0 for v in [*a.values(),*b.values()])
  assert a==dict(created=0,ready=0,pre=86*f,eligible_bf16_t1=t1,completed=0,other_t=86*(f-n),f32_t1=0)
  assert b==dict(observed=mode*86*f,eligible=mode*t1,selected=mode*t1,completed=mode*t1,prefill=mode*86*(f-n),f32_t1=0,layout=0,selected_exceptions=0,fallback_exceptions=0,active=0)
 return proof
