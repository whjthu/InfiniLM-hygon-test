import pathlib,json,sys
q=json.loads(pathlib.Path(sys.argv[1]).read_text());assert q['status']=='ACTUAL_A_OFFLINE_CANONICAL_PROPOSALS_COMPLETE'and q['shutdown_returncode']==0
assert not q['synthetic_features']and not q['incremental_used']and not q['serving_acceptance_claim']and not q['spec_speedup_claim']
assert len(q['ordinary_A_reference_IDs33'])==33 and len(q['cases'])==4
for c,N in zip(q['cases'],[512,520,528,536]):
 assert c['context_tokens']==N and c['anchor_ID']==q['ordinary_A_reference_IDs33'][N-512] and c['ordinary_A_true_next7_IDs']==q['ordinary_A_reference_IDs33'][N-512+1:N-512+8]
 assert c['finite']and c['ownrepeat_proposal_IDs_consistent']and len(c['warm_query_wall_samples_s'])==3 and all(x>0 for x in c['warm_query_wall_samples_s']) and len(c['draft_selector_top1_IDs'])==7 and 0<=c['matched_prefix_0_to7']<=7
print(json.dumps({'passed':True,'scope':'real A offline canonical proposal agreement, not serving acceptance or speedup','prefixes':[x['matched_prefix_0_to7']for x in q['cases']]}))
