import pathlib,json,sys
q=json.loads(pathlib.Path(sys.argv[1]).read_bytes());assert q['completed']and q['normal']and len(q['owners'])==8
for o in q['owners']:
 assert o['normal_owner_closed']and len(o['measurements'])==108 and len(o['guard_rejections'])==12
 assert all(m['finite']and m['canary']and m['readonly']and m['host_profile']is None for m in o['measurements'])
 assert all(g['successful_prepare_state_unchanged']for g in o['guard_rejections'])
 for name in ['continuous_threshold_EOS7','zero_visible_T1','zero_visible_T7']:
  for arm in [0,1]:
   seq=[m for m in o['measurements']if m['fixture']==name and m['arm']==arm];assert all(m['no_reset_since_prefix']for m in seq)
   assert all(b['smallt_after'][4]==a['smallt_after'][4]+1 for a,b in zip(seq,seq[1:]))
   if name=='continuous_threshold_EOS7':
    assert len(seq)==10 and seq[0]['counts']['allselected']==arm and seq[-1]['counts']['fullindexer']==1
    assert seq[5]['rows']==seq[0]['rows'][1:]and any((p+1)//4>512 for p in seq[5]['past'])and any((p+1)//4<=512 for p in seq[5]['past'])
for f in q['fixtures'][:6]:assert len(f['performance']['matched_iterations'])==8
print(json.dumps({'qualified':True,'step':q['step'],'level':'WHOLE_ACTUAL_CSA_C4_SMALLT_COMPONENT_ONLY','Model_gain_claim':False,'directIndexer_abort_stale_foreigncommit':'NOTCOVERED','continuous_threshold_state_and_causalguards':True}))
