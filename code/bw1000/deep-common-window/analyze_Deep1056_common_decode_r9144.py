import pathlib,json,hashlib,statistics,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';p=r/'results/deep-eos-compaction-r9130e-910374.1056.json';raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();assert h=='f69c1a26bd56c6d1880dbc94b739042945747a0a6d2b14d2de259e308e47dc80';d=json.loads(raw)
def describe(v):
 return {'n':len(v),'min':min(v),'median':statistics.median(v),'mean':statistics.mean(v),'max':max(v)}if v else {'n':0}
out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_report_SHA256':h,'actual_handle':'910374.1056','source_status':{'completed':d.get('completed'),'engine_worker_shutdown_completed':d.get('engine_worker_shutdown_completed'),'grid_complete':d.get('grid_complete')},'cards':8,'scope':'Existing1056native actual timestamps reanalysis; no newGPU or optimization round; commondecodewindow distinct from CAPI sum; generated nonEOS includesreasoning/control,notvisibleansweronly; notHTTP','cohorts':[],'grid':{}}
for q in d['quality_trials']:
 windows=q['windows'];pre=[x for x in windows if x['prefill']];dec=[x for x in windows if not x['prefill']];assert pre and dec
 anchor=max(x['return_monotonic']for x in pre);end=max(x['return_monotonic']for x in dec);wall=end-anchor;assert wall>0
 first_call_start=dec[0]['return_monotonic']-dec[0]['seconds'];users=[];useful=0;good_tokens=0
 for a,record in zip(q['requests'],q['records']):
  e=[x for x in a['events']if x['kind']=='committed'and x['token_id']!=1];stamps=[x['time']for x in e];assert stamps==sorted(stamps)
  rate=(len(stamps)-1)/(stamps[-1]-stamps[0])if len(stamps)>1 and stamps[-1]>stamps[0]else None
  nu=sum(x['time']>anchor for x in e);useful+=nu;good=record['stop_reason']=='EOS'and record['score']['flexible_correct'];good_tokens+=nu*good
  users.append({'id':record['id'],'decode_tok_s_firstlast':rate,'generated_nonEOS':len(e),'useful_decode_after_common_anchor':nu,'stop_reason':record['stop_reason'],'taskcorrect':record['score']['flexible_correct'],'TTFT_seconds':stamps[0]-a['arrival']if stamps else None,'ITL_max_seconds':max([b-a for a,b in zip(stamps,stamps[1:])],default=None),'first_ready_to_last_ready_s':stamps[-1]-stamps[0]if len(stamps)>1 else None})
 assert useful==q['native_decode_useful_tokens'],(q['label'],useful,q['native_decode_useful_tokens'])
 out['cohorts'].append({'label':q['label'],'B':q['B'],'compact':q['compact_EOS_rows'],'common_decode_window_s':wall,'CAPI_decode_sum_s':q['native_decode_seconds'],'pre_first_decode_host_gap_s':first_call_start-anchor,'between_CAPI_host_and_validation_s':end-first_call_start-q['native_decode_seconds'],'common_decode_useful_tokens':useful,'common_decode_tok_s':useful/wall,'common_decode_tok_s_per_card':useful/wall/8,'natural_correct_decode_goodput_tok_s':good_tokens/wall,'native_CAPI_decode_tok_s':q['native_decode_useful_tok_s'],'prefill_input_tokens':q['prefill_completed_input_tokens'],'native_prefill_s':q['native_prefill_seconds'],'driver_wall_s':q['driver_wall_seconds'],'users':users,'user_rate_distribution':describe([x['decode_tok_s_firstlast']for x in users if x['decode_tok_s_firstlast']is not None])})
for B in [8,16,32]:
 out['grid'][str(B)]={}
 for compact in [False,True]:
  xs=[x for x in out['cohorts']if x['B']==B and x['compact']==compact];n=sum(x['common_decode_useful_tokens']for x in xs);wall=sum(x['common_decode_window_s']for x in xs);rates=[u['decode_tok_s_firstlast']for x in xs for u in x['users']if u['decode_tok_s_firstlast']is not None]
  out['grid'][str(B)]['compact'if compact else'fixed']={'cohorts':len(xs),'common_decode_tokens':n,'common_decode_window_sum_s':wall,'common_decode_tok_s':n/wall,'common_decode_tok_s_per_card':n/wall/8,'user_rate_distribution':describe(rates),'natural_correct_requests':sum(u['stop_reason']=='EOS'and u['taskcorrect']for x in xs for u in x['users']),'caps':sum(u['stop_reason']!='EOS'for x in xs for u in x['users']),'users_above15':sum(v>15 for v in rates)}
print(json.dumps(out))
