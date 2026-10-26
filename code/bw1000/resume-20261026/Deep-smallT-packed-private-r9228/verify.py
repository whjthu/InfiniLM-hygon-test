import pathlib,json,sys
q=json.loads(pathlib.Path(sys.argv[1]).read_text());assert q['completed']and q['normal']and len(q['fixtures'])==20
for f in q['fixtures']:
 assert f['numerical_contract_passed']and len(f['measurements'])==(3 if f['T']==1 else 11)
 for v in f['measurements']:
  assert v['finite']and v['readonly']and v['canary']
  if f['T']>=8:assert v['fused_calls']==1 and v['fallback_calls']==0 and v['smallT_counts']==[0]*9
  else:assert v['fused_calls']==0 and v['fallback_calls']==1 and v['smallT_counts'][0]==1
print(json.dumps({'qualified':True,'step':q['step'],'level':'ACTUAL_OUTER_LATENT_COMPONENT_ONLY','B32_gain_claim':False}))
