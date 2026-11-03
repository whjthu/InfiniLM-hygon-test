"""Independent passive checks of actual typed Model-owner coverage; no CAPI calls."""
MODES=(0,1,0,1,1,0)
def validate(row,index):
 assert type(index)is int and 0<=index<6
 mode=MODES[index];assert row['r438_index']==index and row['r438_mode']==mode and row['r438_warm']==(index<2)
 assert row['r438_quiescent'] is True and row['r438_poisoned'] is False
 assert row['normal_reset_before'] and row['normal_reset_after'] and not row.get('error') and not row.get('reset_error')
 f=row['actual_forward_completed'];assert type(f)is int and f>0 and f==row['actual_forward_attempts']
 trace=row['Native_continuation'];prefill=[x for x in trace if x['kind']=='prefill'];decode=[x for x in trace if x['kind']=='decode_summary']
 assert len(prefill)==9 and [x['past']for x in prefill]==list(range(0,8489,1024)) and [x['tokens']for x in prefill]==[1024]*8+[297]
 assert [x['total']for x in prefill]==[x['past']+x['tokens']for x in prefill]
 n=sum(x['calls']for x in decode);assert len(decode)<=1 and f==9+n and len(row['generated_token_ids'])==n+1
 if decode:assert decode[0]['first_past']==8489 and decode[0]['last_total']==8489+n
 t1=43*n;proof=row['r438_actual_T1_coverage']
 assert proof==dict(passed=True,MoE_layers=43,completed_forwards=f,one_token_prefill_forwards=0,decode_forwards=n,T1_per_owner=t1,mode=mode)
 assert t1>0
 counts=row['r438_actual_owner_counts'];assert len(counts)==8
 for c in counts:
  assert all(type(v)is int and v>=0 for v in c.values())
  assert c['entered_valid_MoE']==c['host_returned_not_GPU_completed']==43*f
  assert c['T1']==t1 and c['armed_mode1_T1']==mode*t1 and c['last_mode']==mode and c['unwound']==0
  for role in ['routed','shared']:
   assert c[role+'_hits']==mode*t1 and c[role+'_calls']==c[role+'_hits']+c[role+'_fallback']
 return proof
