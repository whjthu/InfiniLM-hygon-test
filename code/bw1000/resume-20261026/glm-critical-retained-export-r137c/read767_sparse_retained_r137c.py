"""Only existing767 metadata and sparse marks; never executes Model/Runtime/GPU."""
import pathlib,json,hashlib,math
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
AUDIT_SHA='814ad28995f2172fe719c2c9a8facfbfb1e3f4626dec3af3b26649381c4ee164'
FILE_LIMIT=64*1024*1024;TOTAL_LIMIT=512*1024*1024
out={'scope':'exact767 retained16owner sparse events and pipeline6/sequence/row-past-count metadata only','audit_SHA256':AUDIT_SHA,'read_limits':{'per_file_bytes':FILE_LIMIT,'total_bytes':TOTAL_LIMIT},'owners':[],'raw_tokens_weights_activations_logs_exported':False}
diagnostic=[];total=0
try:
 audit=r/'results/GLM-critical-Model-r133-audit.json';araw=audit.read_bytes();ah=hashlib.sha256(araw).hexdigest()
 if ah!=AUDIT_SHA:raise ValueError(('AUDIT_SHA_MISMATCH_STOP',ah,AUDIT_SHA))
 a=json.loads(araw);owners=a['owners'];out['run']=a['run']
 if len(owners)!=16 or {(o['stage'],o['rank'])for o in owners}!={(s,k)for s in [0,1]for k in range(8)}:raise ValueError('AUDIT_OWNER_SET_STOP')
 for o in owners:
  stage=o['stage'];rank=o['rank'];p=r/'results'/f"glm-arrival-r133-{a['run']}-stage{stage}-rank{rank}.json";st=p.stat();meta={'stage':stage,'rank':rank,'source_path':str(p),'bytes':st.st_size,'expected_SHA256':o['SHA256'],'expected_producer':o['actual_step']};diagnostic.append(meta)
  if st.st_size>FILE_LIMIT or total+st.st_size>TOTAL_LIMIT:raise ValueError('READ_SIZE_BOUND_STOP')
  raw=p.read_bytes();total+=len(raw)
  if len(raw)>FILE_LIMIT or total>TOTAL_LIMIT:raise ValueError('FILE_CHANGED_OVER_READ_BOUND_STOP')
  h=hashlib.sha256(raw).hexdigest();meta['actual_SHA256']=h
  if h!=o['SHA256']:raise ValueError('RESULT_SHA_MISMATCH_STOP')
  q=json.loads(raw);producer=q.get('actual_producer');meta['actual_producer']=producer if isinstance(producer,str)else{'type':type(producer).__name__}
  if producer!=o['actual_step']or producer!='910376.767':raise ValueError('PRODUCER_MISMATCH_STOP')
  episodes=[]
  for e in q['episodes']:
   events=[]
   for i,c in enumerate(e['native_calls']):
    v=q['native_events'][c['event_index']];values=v['existing_pipeline_profile']
    if len(values)!=6 or not all(isinstance(x,(int,float))and math.isfinite(x)and x>=0 for x in values):raise ValueError('PIPELINE6_INVALID_STOP')
    events.append({'owner_sequence':i+1,'command_sequence':c['seq'],'op':c['op'],'rows':c['rows'],'pasts':c['pasts'],'counts':c['counts'],'tokens':v['input_tokens'],'native_seconds':v['native_seconds'],'pipeline6':values})
   s=e['sparse_observer'];sparse={k:s[k]for k in ['coverage','enabled','events_allocated','missing_is_UNKNOWN','required','sequence']};sparse['samples']=[]
   for sample in s['samples']:
    item={k:sample[k]for k in ['clock_scope','event_count','full_detail_layer','shared_detail_layer','rank','stage','row_past_count','sequence','tokens']}
    if sample['stage']!=stage or sample['rank']!=rank or sample['event_count']!=len(sample['marks'])or len(sample['marks'])>512:raise ValueError('SPARSE_OWNER_BOUND_STOP')
    item['marks']=[{k:m[k]for k in ['name','owner_host_s','gpu_from_first_s']}for m in sample['marks']];sparse['samples'].append(item)
   episodes.append({'label':e['label'],'sparse_observer':sparse,'events':events})
  out['owners'].append({'stage':stage,'rank':rank,'source_path':str(p),'source_SHA256':h,'source_bytes':len(raw),'producer':producer,'episodes':episodes})
 out.update(completed=True,total_source_bytes_read=total,source_owner_count=16,classification='ALL16_SHA_PRODUCER_BOUND_EXPORT_ONLY_EXISTING_SPARSE_SCALARS',old_failure_clarification='stage1rank0 10064251bytes exceeds old10000000limit only; SHA/producer gates unchanged')
 encoded=json.dumps(out,separators=(',',':'))
 if len(encoded.encode('utf8'))>32*1024*1024:raise ValueError('EXPORT_SIZE_BOUND_STOP')
 print(encoded)
except Exception as e:
 print(json.dumps({'completed':False,'scope':out['scope'],'audit_SHA256':AUDIT_SHA,'error_type':type(e).__name__,'classification':str(e),'diagnostic':diagnostic,'owners_exported':0,'no_source_identity_gate_relaxed':True},separators=(',',':')))
