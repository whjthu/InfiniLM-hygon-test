import pathlib,json,hashlib,statistics,collections,argparse
ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);opt=ap.parse_args();src=pathlib.Path(opt.input);p=pathlib.Path(__file__).resolve().parent;a=json.loads(src.read_bytes());assert a['completed']and a['source_owner_count']==16 and a['audit_SHA256']=='814ad28995f2172fe719c2c9a8facfbfb1e3f4626dec3af3b26649381c4ee164'
def describe(v):return {'min':min(v),'median':statistics.median(v),'max':max(v),'n':len(v)}
samples=[];full=[];aggregates=collections.defaultdict(list)
for owner in a['owners']:
 stage=owner['stage'];rank=owner['rank'];assert owner['producer']=='910376.767';e=next(x for x in owner['episodes']if x['label']=='sparse-profile');sparse=e['sparse_observer'];assert sparse['coverage']==[3,3,1,1]
 for sample in sparse['samples']:
  seq=sample['sequence'];event=e['events'][seq-1];assert event['owner_sequence']==seq and sample['row_past_count']==[list(x)for x in zip(event['rows'],event['pasts'],event['counts'])]and sample['tokens']==event['tokens'];marks=sample['marks'];by={m['name']:m for m in marks}
  def gpu(x,y):return 1000*(by[y]['gpu_from_first_s']-by[x]['gpu_from_first_s'])
  def host(x,y):return 1000*(by[y]['owner_host_s']-by[x]['owner_host_s'])
  layers=[]
  for m in marks:
   if m['name'].startswith('layer.')and m['name'].endswith('.begin'):
    l=int(m['name'].split('.')[1]);layers.append({'layer':l,'type':m['name'].split('.')[2],'GPUstream_ms':gpu(m['name'],f'layer.{l}.end'),'HOSTspan_ms':host(m['name'],f'layer.{l}.end')})
  kind=('T'+str(event['tokens']))if event['op']=='DECODE'else'B'+str(len(event['rows']))+' PREFIX'
  v={'stage':stage,'rank':rank,'sequence':seq,'kind':kind,'command_sequence':event['command_sequence'],'selector_SHA256':hashlib.sha256(json.dumps(sample['row_past_count'],separators=(',',':')).encode()).hexdigest(),'native_CAPI_ms':event['native_seconds']*1000,'pipeline6_ms':[x*1000 for x in event['pipeline6']],'owner_GPUstream_ms':gpu('CAPI.owner.begin','CAPI.owner.end.before_original_drain'),'PP_incoming_GPU_ms':gpu('PP.incoming.begin','PP.incoming.end'),'PP_outgoing_GPU_ms':gpu('PP.outgoing.begin','PP.outgoing.end'),'actual_distinct_layer_GPU_sum_ms':sum(x['GPUstream_ms']for x in layers),'largest_single_actual_layer_GPU_ms':max(x['GPUstream_ms']for x in layers),'layers':layers,'detail':[]}
  for l in [sample['full_detail_layer'],sample['shared_detail_layer']]:
   d={'layer':l,'type':'full'if l==sample['full_detail_layer']else'shared'}
   for phase in ['RMS.attention','attention','RMS.ffn','FFN']:
    pre=f'detail.layer.{l}.{phase}';d[phase+'_GPU_ms']=gpu(pre+'.begin',pre+'.end');d[phase+'_HOST_ms']=host(pre+'.begin',pre+'.end')
   v['detail'].append(d)
  samples.append(v);aggregates[(stage,seq,kind,v['selector_SHA256'])].append(v)
 for ep in owner['episodes']:
  sums=collections.defaultdict(lambda:{'calls':0,'CAPI_s':0.,'pipeline6_sum_s':[0.]*6})
  for ev in ep['events']:
   d=sums[ev['op']];d['calls']+=1;d['CAPI_s']+=ev['native_seconds'];d['pipeline6_sum_s']=[x+y for x,y in zip(d['pipeline6_sum_s'],ev['pipeline6'])]
  full.append({'stage':stage,'rank':rank,'label':ep['label'],'operation':dict(sums)})
rows=[]
for (stage,seq,kind,selector),vs in sorted(aggregates.items()):
 assert sorted(v['rank']for v in vs)==list(range(8));r={'stage':stage,'sequence':seq,'kind':kind,'selector_SHA256':selector,'rank_count':8}
 for field in ['native_CAPI_ms','owner_GPUstream_ms','PP_incoming_GPU_ms','PP_outgoing_GPU_ms','actual_distinct_layer_GPU_sum_ms','largest_single_actual_layer_GPU_ms']:r[field]=describe([v[field]for v in vs])
 r['detail']=[{'type':vs[0]['detail'][i]['type'],'layer':vs[0]['detail'][i]['layer'],**{k:describe([v['detail'][i][k]for v in vs])for k in ['attention_GPU_ms','attention_HOST_ms','FFN_GPU_ms','FFN_HOST_ms']}}for i in [0,1]];rows.append(r)
for seq in {x['sequence']for x in samples}:assert len({x['selector_SHA256']for x in samples if x['sequence']==seq})==1
out={'schema':1,'source_SHA256':hashlib.sha256(src.read_bytes()).hexdigest(),'audit_SHA256':a['audit_SHA256'],'all16_identity_verified':True,'owners':16,'sparse_samples':len(samples),'all16_coverage':[3,3,1,1],'stage_local_same_selector_rank_duration_distributions':rows,'actual_samples':samples,'full_episode_owner_pipeline6':full,'interpretation':['Compare durations only for same seq/full selector; no crosshost clock subtraction','Do not add stage0+stage1, or HOST+GPU nested spans','Actual distinct layer GPU sum is same-owner disjoint observed intervals; never multiply representative detail layers','Internal CCL time/expert job skew/physical BWpeak remain UNKNOWN'],'source_patch_count':0}
(p/'all16-critical-attribution-r137c.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps({'all16_identity_verified':True,'sparse_samples':len(samples),'rows':rows},indent=2))
