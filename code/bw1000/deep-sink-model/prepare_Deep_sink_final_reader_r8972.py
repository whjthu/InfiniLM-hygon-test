import pathlib,ast
r=pathlib.Path(__file__).resolve().parent;s=(r/'read_Deep_sink_Model_watch_r8971.py').read_text();needle='print(json.dumps(d))';assert s.count(needle)==1
s=s.replace(needle,'''if 'q' in locals():
 d['actual_counter_receipts']=[]
 for t in q.get('owner_trials_r8970',[]):
  row={'label':t['label'],'sink_arm':t.get('sink_arm'),'complete':t['complete'],'whole_scratch_ranges':[[min(v),max(v)]for v in zip(*t.get('scratch_counter_delta',[]))],'whole_sink_ranges':[[min(v),max(v)]for v in zip(*t.get('sink_counter_delta',[]))],'decode_fused_counts':t.get('accepted_decode_fused_counter_delta'),'failure':t.get('failure')}
  row['chunks']=[{'chunk':x['chunk'],'T':x['total_tokens'],'scratch_ranges':[[min(v),max(v)]for v in zip(*x['counter_delta'])],'sink_ranges':[[min(v),max(v)]for v in zip(*x.get('sink_counter_delta',[]))]}for x in t.get('scratch_prefill_chunk_checks',[])]
  d['actual_counter_receipts'].append(row)
print(json.dumps(d))''');ast.parse(s);(r/'read_Deep_sink_Model_final_r8972.py').write_bytes(s.encode())
