"""Passive actual variable-prefix86HC proof; wrong/short answers are not filtered."""
def validate(row,entry):
 mode=entry['mode'];assert row['r450_logical_index']==entry['logical_index']and row['r450_case_id']==entry['case_id']and row['r450_mode']==mode and row['r450_r366_fixed']==0
 assert row['normal_reset_before']and row['normal_reset_after']and row['r450_quiescent']is True and row['r450_poisoned']is False
 assert not row.get('error')and not row.get('reset_error')
 f=row['actual_forward_completed'];assert type(f)is int and f>0 and f==row['actual_forward_attempts']
 ids=row['generated_token_ids'];assert ids and len(ids)<=4096 and row['max_new_tokens']==4096 and row['prompt_tokens']==entry['prompt_tokens']
 p=entry['prompt_tokens'];pre=[v for v in row['Native_continuation']if v['kind']=='prefill'];ds=[v for v in row['Native_continuation']if v['kind']=='decode_summary']
 assert [v['past']for v in pre]==list(range(0,p,1024))and[v['tokens']for v in pre]==[min(1024,p-i)for i in range(0,p,1024)]and all(v['total']==v['past']+v['tokens']for v in pre)
 decode=len(ids)-1;assert f==len(pre)+decode
 assert ds==([dict(kind='decode_summary',calls=decode,first_past=p,last_total=p+decode)]if decode else[])
 one=sum(v['tokens']==1 for v in pre);t1=86*(one+decode)
 counts=row['r450_actual_owner_counts'];assert len(counts)==8
 for d,c in enumerate(counts):
  assert c['owner']==d and c['created']==c['ready']==86
  a,b=c['legacy'],c['fullpre'];assert all(type(v)is int and v>=0 for v in [*a.values(),*b.values()])
  assert a==dict(created=0,ready=0,pre=86*f,eligible_bf16_t1=t1,completed=0,other_t=86*(f-one-decode),f32_t1=0)
  assert b==dict(observed=mode*86*f,eligible=mode*t1,selected=mode*t1,completed=mode*t1,prefill=mode*86*(f-one-decode),f32_t1=0,layout=0,selected_exceptions=0,fallback_exceptions=0,active=0)
 return dict(passed=True,HC_groups=86,completed_forwards=f,prefill_forwards=len(pre),one_token_prefill_forwards=one,decode_forwards=decode,T1_HC_per_owner=t1,mode=mode,r366_fixed=0,no_eligible_T1=t1==0)
