"""R169 owner-local event/host intervals. No GPU-busy or exclusive TP claim."""
import math,statistics

def consume(observer,native_calls,native_events):
 if observer.get('selector_revision')!='r169' or observer.get('required')!=[0,0,0,0,3]:raise ValueError('wrong selector contract')
 if observer.get('coverage')!=[0,0,0,0,3] or len(observer.get('samples',[]))!=3:return {'coverage':'UNKNOWN','reason':'three eligible T1 windows absent','TP_only':'UNKNOWN'}
 if [s['eligible_decode1_index']for s in observer['samples']]!=[17,18,19]:raise ValueError('duplicate or reordered T1 windows')
 byseq={i+1:c for i,c in enumerate(native_calls)};windows=[]
 for sample in observer['samples']:
  if sample['selector']!='decode_T1' or sample['tokens']!=1 or len(sample['row_past_count'])!=1 or sample['eligible_decode1_index']not in [17,18,19]:raise ValueError('sample not selected T1')
  call=byseq[sample['sequence']];event=native_events[call['event_index']]
  if call['op']!='DECODE' or event['prefill'] or call['counts']!=[1] or call['pasts'][0]<=0 or sample['row_past_count']!=[[call['rows'][0],call['pasts'][0],1]]:raise ValueError('anchor/prefix cannot qualify decode')
  marks=sample['marks'];names=[m['name']for m in marks]
  if len(set(names))!=len(names) or len(marks)>512 or names[0]!='CAPI.owner.begin' or names[-1]!='CAPI.owner.end.before_original_drain':raise ValueError('invalid event tape')
  for field in ['owner_host_s','gpu_from_first_s']:
   values=[m[field]for m in marks]
   if any(not math.isfinite(x)for x in values)or values!=sorted(values):raise ValueError('invalid clock')
  points={m['name']:m for m in marks}
  def span(a,b):
   x,y=points[a],points[b]
   return {'stream_interval_ms':1000*(y['gpu_from_first_s']-x['gpu_from_first_s']),'host_span_ms':1000*(y['owner_host_s']-x['owner_host_s'])}
  layers=[];details=[]
  for name in names:
   if name.startswith('layer.')and name.endswith('.begin'):
    layer=int(name.split('.')[1]);layers.append(dict(layer=layer,kind=name.split('.')[2],**span(name,'layer.'+str(layer)+'.end')))
   if name.startswith('detail.layer.')and name.endswith('.begin')and name[:-6]+'.end'in points:
    details.append(dict(name=name[:-6],**span(name,name[:-6]+'.end')))
  gaps=sorted([{'from':x['name'],'to':y['name'],'host_enqueue_spacing_ms':1000*(y['owner_host_s']-x['owner_host_s']),'stream_interval_ms':1000*(y['gpu_from_first_s']-x['gpu_from_first_s'])}for x,y in zip(marks,marks[1:])],key=lambda x:x['host_enqueue_spacing_ms'],reverse=True)[:8]
  windows.append({'sequence':sample['sequence'],'eligible_T1_index':sample['eligible_decode1_index'],'row':call['rows'][0],'past':call['pasts'][0],'actual_op':'DECODE','owner_CAPI':span(names[0],names[-1]),'PP_incoming':span('PP.incoming.begin','PP.incoming.end'),'PP_outgoing':span('PP.outgoing.begin','PP.outgoing.end'),'layers':layers,'nested_detail_not_additive':details,'largest_host_enqueue_spacings':gaps,'original_forward_HOST_CAPI_ms':1000*event['native_seconds']})
 return {'coverage':'COMPLETE','windows':windows,'TP_only':'UNKNOWN','GPU_busy':'UNKNOWN','host_and_device_spans_overlap':True,'stream_buckets_include_gaps_and_dependencies':True,'do_not_sum_nested_detail_with_layers_or_host':True,'clock_scope':'one original owner stream; no cross-owner timestamp subtraction'}

def overhead(episodes,native_events):
 """Compare adjacent OFF/ON/OFF selected complete Native calls; profile arm is not clean TPS."""
 medians=[];all_episode=[]
 for arm in episodes:
  eligible=[]
  for call in arm['native_calls']:
   event=native_events[call['event_index']]
   if len(call['rows'])==1 and call['counts']==[1]and call['pasts'][0]>0:eligible.append((call,event))
  chosen=eligible[16:19]
  if len(chosen)!=3 or any(c['op']!='DECODE' or e['prefill']for c,e in chosen):return {'status':'UNKNOWN','reason':'matching actual decode windows unavailable'}
  medians.append(statistics.median(e['native_seconds']for c,e in chosen))
  events=[native_events[c['event_index']]for c in arm['native_calls']if c['op']=='DECODE']
  all_episode.append({'label':arm['label'],'observer':arm['observer_enabled'],'decode_calls':len(events),'decode_HOST_CAPI_seconds':sum(e['native_seconds']for e in events)})
 before,on,after=medians;base=(before+after)/2
 drift=100*abs(after/before-1);penalty=100*(on/base-1)
 return {'status':'MEASURED','matching_window_median_HOST_CAPI_ms':[1000*x for x in medians],'OFF_endpoint_drift_percent':drift,'ON_vs_meanOFF_penalty_percent':penalty,'bounded_overhead_screen_passed':drift<=3 and penalty<=3,'thresholds_percent':{'max_OFF_drift':3,'max_ON_penalty':3},'episodes':all_episode,'ON_not_clean_TPS':True,'token_counts_and_quality_reported_separately':True,'one_triple_no_significance_claim':True}
