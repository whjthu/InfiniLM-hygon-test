"""Frozen512 logical OFF/ON requests; no model topology or time-grid choice."""
def make(cases,encodings):
 assert len(cases)==len(encodings)==256 and len({c['id']for c in cases})==256
 assert [c['id']for c in cases]==[e['case_id']for e in encodings]
 assert sum(c['suite']=='gsm8k'for c in cases)==sum(c['suite']=='mmlu'for c in cases)==128
 rows=[]
 for i,(c,e)in enumerate(zip(cases,encodings)):
  assert e['suite']==c['suite']and e['output_budget']==4096 and 0<e['prompt_tokens']<=32768-4096
  for mode in ([0,1]if i%2==0 else[1,0]):
   rows.append(dict(logical_index=len(rows),case_id=c['id'],case_ordinal=i,mode=mode,request_body_SHA256=e['request_body_SHA256'],prompt_IDs_SHA256=e['prompt_IDs_SHA256'],prompt_tokens=e['prompt_tokens'],output_budget=4096))
 assert len(rows)==512 and sum(r['mode']==0 for r in rows)==sum(r['mode']==1 for r in rows)==256
 return rows

def validate_subset(rows,frozen):
 assert rows and len(rows)<=512 and len({r['logical_index']for r in rows})==len(rows)
 lookup={r['logical_index']:r for r in frozen};assert len(lookup)==512
 for row in rows:assert row==lookup[row['logical_index']],'Foreign/modified finite plan entry'
 return rows
