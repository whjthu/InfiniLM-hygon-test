import pathlib,ast
r=pathlib.Path.cwd();s=(r/'read_Deep_scratch_Model_watch_r8956.py').read_text();needle='print(json.dumps(d))';assert s.count(needle)==1
s=s.replace(needle,'''if 'q' in locals():
 d['actual_scratch_counter_receipts']=[]
 for t in q.get('owner_trials_r8956',[]):
  row={'label':t['label'],'scratch_arm':t.get('scratch_arm'),'complete':t['complete'],'prefill_chunk_count':len(t.get('scratch_prefill_chunk_checks',[])),'whole_counter_ranges':[[min(v),max(v)]for v in zip(*t.get('scratch_counter_delta',[]))],'decode_fused_counts':t.get('accepted_decode_fused_counter_delta'),'failure':t.get('failure')}
  chunks=t.get('scratch_prefill_chunk_checks',[])
  row['prefill_chunk_all8_counter_ranges']=[{'chunk':x['chunk'],'T':x['total_tokens'],'ranges':[[min(v),max(v)]for v in zip(*x['counter_delta'])]}for x in chunks]
  d['actual_scratch_counter_receipts'].append(row)
print(json.dumps(d))''');ast.parse(s);(r/'read_Deep_scratch_Model_final_r8957.py').write_bytes(s.encode())
s=(r/'summarize_Deep_latent_Model_r8944.py').read_text().replace("t.get('latent_arm')","t.get('scratch_arm')").replace('12ed7e404072b4f2c11a4337d0c566225afdb15f','622cbf5e8d430434b09139314f6b7b1c069212b2').replace('defaultOFF prefillqueryT32tiledexistingr681kernel','defaultOFF FP32reference scratchreuse;original8933prefilltilesOFF').replace('deep-latent-prefill-model-cpu-ready-r8943','deep-reference-scratch-model-cpu-ready-r8956').replace('4b35d906093b227cce1e62147d80efdb051c28d6b5a4408f03831fd387ec9cbc','68c1185f23debd7a74f0c9bae1613af2d271bc2ef06453f7957e9cacfb6c75ed').replace('deep908-prefill-attention-tiles-model','deep923-reference-attention-scratch-model')
s=s.replace("component_comparison={'handle':'910374.905','maxowner_A_ms':298.26851189136505,'maxowner_C_ms':20.529824192635715,'A_over_C':14.528546815240503,'scope':'Syntheticattention operands only,8owner/streamsamplesnot8independentdata. NeverusecomponentfactorasModelgain.'}","component_comparison={'handle':'910374.920','scope':'Syntheticlatent-referenceattention scratch componentclosingpair15..17percentonly;firstA0anomalyanddriftretained,notModelprediction.'},actual_scratch_counter_receipts=d.get('actual_scratch_counter_receipts')")
ast.parse(s);(r/'summarize_Deep923_r8957.py').write_bytes(s.encode());print('CPU final reader/reducer prepared; not executed beforeoriginalterminal+post')
