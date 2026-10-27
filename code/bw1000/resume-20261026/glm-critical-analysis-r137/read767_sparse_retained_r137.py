import pathlib,json,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');a=json.loads((r/'results/GLM-critical-Model-r133-audit.json').read_bytes());out={'scope':'exact767 retained16owner sparse/pipeline scalar metadata only; no tokens/weights/logs','run':a['run'],'owners':[]}
for o in a['owners']:
 stage=o['stage'];rank=o['rank'];p=r/'results'/f"glm-arrival-r133-{a['run']}-stage{stage}-rank{rank}.json";raw=p.read_bytes();assert len(raw)<10000000;h=hashlib.sha256(raw).hexdigest();assert h==o['SHA256'];q=json.loads(raw);assert q['actual_producer']=='910376.767'
 episodes=[]
 for e in q['episodes']:
  events=[]
  for c in e['native_calls']:
   v=q['native_events'][c['event_index']];events.append({'owner_sequence':len(events)+1,'command_sequence':c['seq'],'op':c['op'],'rows':c['rows'],'pasts':c['pasts'],'counts':c['counts'],'tokens':v['input_tokens'],'native_seconds':v['native_seconds'],'pipeline6':v['existing_pipeline_profile']})
  episodes.append({'label':e['label'],'sparse_observer':e['sparse_observer'],'events':events})
 out['owners'].append({'stage':stage,'rank':rank,'source_path':str(p),'source_SHA256':h,'episodes':episodes})
assert len(out['owners'])==16
print(json.dumps(out,separators=(',',':')))
