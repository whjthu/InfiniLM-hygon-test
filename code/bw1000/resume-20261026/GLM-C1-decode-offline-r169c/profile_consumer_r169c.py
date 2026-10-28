"""R169 owner-local event/host intervals. No GPU-busy or exclusive TP claim."""
import math,statistics
import re,collections

def ordered_tape_r169c(marks):
 """Preserve every repeated projection occurrence; structure is strict and LIFO."""
 names=[m['name']for m in marks];counts=collections.Counter(names)
 structural=['CAPI.owner.begin','CAPI.owner.end.before_original_drain','PP.incoming.begin','PP.incoming.end','PP.outgoing.begin','PP.outgoing.end']
 if len(marks)>512 or names[0]!=structural[0]or names[-1]!=structural[1]or any(counts[x]!=1 for x in structural):raise ValueError('structural endpoint multiplicity')
 points={name:marks[names.index(name)]for name in structural};stack=[];seen_layers=set();layers=[];details=[];projections=[];pending={};layer=None;phase='CAPI';occ=collections.Counter()
 def span(x,y):return {'stream_interval_ms':1000*(y['gpu_from_first_s']-x['gpu_from_first_s']),'host_span_ms':1000*(y['owner_host_s']-x['owner_host_s'])}
 for index,m in enumerate(marks):
  name=m['name']
  if name=='CAPI.owner.begin':
   if index!=0 or stack:raise ValueError('CAPI begin order')
   stack.append(('CAPI',m));continue
  if name=='CAPI.owner.end.before_original_drain':
   if index!=len(marks)-1 or len(stack)!=1 or stack[-1][0]!='CAPI'or phase!='done'or pending:raise ValueError('CAPI close order')
   stack.pop();continue
  if name=='PP.incoming.begin':
   if phase!='CAPI'or len(stack)!=1:raise ValueError('incoming order')
   stack.append(('PP.incoming',m));phase='incoming';continue
  if name=='PP.incoming.end':
   if phase!='incoming'or stack[-1][0]!='PP.incoming':raise ValueError('incoming close order')
   stack.pop();phase='layers';continue
  if name=='PP.outgoing.begin':
   if phase!='layers'or len(stack)!=1 or layer is not None or pending:raise ValueError('outgoing order')
   stack.append(('PP.outgoing',m));phase='outgoing';continue
  if name=='PP.outgoing.end':
   if phase!='outgoing'or stack[-1][0]!='PP.outgoing':raise ValueError('outgoing close order')
   stack.pop();phase='done';continue
  mb=re.fullmatch(r'layer\.([0-9]+)\.(full|shared)\.begin',name)
  if mb:
   number=int(mb[1])
   if phase!='layers'or len(stack)!=1 or layer is not None or number in seen_layers:raise ValueError('duplicate/nested layer begin')
   seen_layers.add(number);layer=number;stack.append(('layer',number,mb[2],m));continue
  me=re.fullmatch(r'layer\.([0-9]+)\.end',name)
  if me:
   number=int(me[1])
   if layer!=number or stack[-1][0]!='layer'or pending:raise ValueError('layer close order')
   _,n,kind,start=stack.pop();layers.append(dict(layer=n,kind=kind,**span(start,m)));layer=None;continue
  md=re.fullmatch(r'(detail\.layer\.([0-9]+)\..+)\.(begin|end)',name)
  if md:
   key=md[1]
   if layer!=int(md[2])or phase!='layers':raise ValueError('detail outside matching layer')
   if md[3]=='begin':
    if any(s[0]=='detail'and s[1]==key for s in stack):raise ValueError('reentered same detail scope')
    occ[key]+=1;stack.append(('detail',key,occ[key],m));continue
   if stack[-1][0]!='detail'or stack[-1][1]!=key:raise ValueError('missing/crossed detail end')
   _,_,ordinal,start=stack.pop();details.append(dict(name=key,occurrence=ordinal,**span(start,m)))
   if key.endswith('.quant')and '.prepare.projection.'in key:
    pkey=key[:-6]
    if pkey in pending:raise ValueError('projection end not consumed before next quant')
    pending[pkey]=(ordinal,start,m)
   continue
  mf=re.fullmatch(r'(detail\.layer\.([0-9]+)\.prepare\.projection\.N[0-9]+\.K[0-9]+)\.projection\.end\.before_original_fence',name)
  if mf:
   key=mf[1]
   if layer!=int(mf[2])or key not in pending:raise ValueError('orphan/duplicate projection fence marker')
   ordinal,start,quant_end=pending.pop(key)
   projections.append({'name':key,'occurrence':ordinal,'quant':span(start,quant_end),'after_quant_to_before_original_fence':span(quant_end,m),'quant_plus_projection':span(start,m)})
   continue
  raise ValueError('unknown event marker')
 if stack or layer is not None or pending:raise ValueError('unclosed event tape')
 return points,layers,details,projections,dict(counts)



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
  if not marks:raise ValueError('empty event tape')
  for field in ['owner_host_s','gpu_from_first_s']:
   values=[m[field]for m in marks]
   if any(not math.isfinite(x)for x in values)or values!=sorted(values):raise ValueError('invalid clock')
  points,layers,details,projections,point_counts=ordered_tape_r169c(marks)
  def span(a,b):
   x,y=points[a],points[b]
   return {'stream_interval_ms':1000*(y['gpu_from_first_s']-x['gpu_from_first_s']),'host_span_ms':1000*(y['owner_host_s']-x['owner_host_s'])}
  gaps=sorted([{'from':x['name'],'to':y['name'],'host_enqueue_spacing_ms':1000*(y['owner_host_s']-x['owner_host_s']),'stream_interval_ms':1000*(y['gpu_from_first_s']-x['gpu_from_first_s'])}for x,y in zip(marks,marks[1:])],key=lambda x:x['host_enqueue_spacing_ms'],reverse=True)[:8]
  windows.append({'sequence':sample['sequence'],'eligible_T1_index':sample['eligible_decode1_index'],'row':call['rows'][0],'past':call['pasts'][0],'actual_op':'DECODE','owner_CAPI':span(names[0],names[-1]),'PP_incoming':span('PP.incoming.begin','PP.incoming.end'),'PP_outgoing':span('PP.outgoing.begin','PP.outgoing.end'),'layers':layers,'nested_detail_not_additive':details,'projection_occurrences_not_additive':projections,'all_marker_occurrence_counts':point_counts,'largest_host_enqueue_spacings':gaps,'original_forward_HOST_CAPI_ms':1000*event['native_seconds']})
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
