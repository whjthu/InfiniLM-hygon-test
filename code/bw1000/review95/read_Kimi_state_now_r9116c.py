import pathlib,json,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');p=r/'Kimi-state-operator-r9116c.json';d=json.loads(p.read_text());last=json.loads(pathlib.Path(d['last_snapshot']).read_text())if d.get('last_snapshot')else{};q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'operator':{k:v for k,v in d.items()if k not in ['events','quality_staged','counter_fields']},'event_count':len(d.get('events',[])),'staged_count':len(d.get('quality_staged',[])),'snapshotUTC':last.get('UTC'),'owners':last.get('owners'),'watch':last.get('watch'),'pilot_counts':last.get('pilot_counts'),'cases':[],'state_partial_evidence':[]}
for c in last.get('cases',[]):
 v={k:c.get(k)for k in ['label','strategy','K','generated_tokens','useful_decode_tokens','decode_user_wall_s','decode_user_tok_s','prefill_local_wall_s','EOS','truncated','mean_useful_outputs_per_cycle','proposal_acceptance','adaptive_fallback_cycle','state_only_replay_cycles','cost_s']};qual=c.get('quality');v['quality']={k:qual.get(k)for k in ['sample_index','expected_answer','extracted_answer','correct','truncated']}if qual else None;q['cases'].append(v)
for c in (last.get('transaction_evidence')or{}).get('cases',[]):
 if c.get('first_state_partial_then_next_forward'):q['state_partial_evidence'].append(c)
print(json.dumps(q))
