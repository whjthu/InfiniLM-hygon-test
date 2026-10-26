import pathlib,json,hashlib,collections,datetime
p=pathlib.Path.home()/'hygon-inference-refactor-20261014/results/Kimi-adaptive-A-r9225-kimi-A-r9234-b493e8b743588186-node2.json'
b=p.read_bytes();h=hashlib.sha256(b).hexdigest();assert h=='64f6f09984718dc5a55820b886f44986db60362b31e15f7bec2a910f48a4cf1f'
q=json.loads(b);out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'source_report_path':str(p),'source_report_SHA256':h,'groups':{}}
for C in [1,2,8,16]:
 for strategy in ['ar','state']:
  cs=[c for c in q['batch_cases']if c['label'].startswith('Anew32-')and c['configured_C']==C and c['strategy']==strategy and not c['adaptive_width']]
  cy=[x for c in cs for x in c['cycles']];us=[u for c in cs for u in c['request_metrics']];uc=[x for u in us for x in u['cycles']];cost={k:sum(x.get(k,0)for x in cy)for k in ['draft_RPC_group_s','verify_CAPI_maxhost_s','replay_CAPI_maxhost_s','cycle_local_wall_s']}
  out['groups'][f'C{C}-{strategy}']={'cohorts':len(cs),'requests':len(us),'configured_widths':dict(collections.Counter(c['configured_width']for c in cs)),'global_cycles':len(cy),'request_cycles':len(uc),'proposed':sum(x['proposed']for x in uc),'accepted':sum(x['accepted_proposals']for x in uc),'useful_emitted_cycles':sum(x['useful_emitted']for x in uc),'verified_rows':sum(x['flattened_T']for x in cy),'actual_active_C_distribution':dict(collections.Counter(x['actual_active_C']for x in cy)),'target_T_distribution':dict(collections.Counter(x['flattened_T']for x in cy)),'cost_s':cost,'tokens':sum(c['total_useful_decode_tokens']for c in cs),'decode_common_s':sum(c['decode_common_window_s']for c in cs)}
print(json.dumps(out))
