"""CPU-only actual-owner event consumer; never infer active kernels from stream spans."""
import pathlib,json,hashlib,statistics,sys,math
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';names=['forward.begin','router.end','down_projection.end','provider.prepare.end','mask.end','provider.pack.begin','provider.pack.end','provider.jobs.end','provider.W13_fused_activation.end','provider.W2_fused_scatter.end','provider.end','TP_reduce.end','norm.end','up_projection.end','shared.end','add.end'];plan=[0,1,0,1,1,0,0,1,1,0]
def analyze(reports):
 rows=[]
 for node,q in enumerate(reports):
  assert q['completed']and q['normal_CUDA_cleanup']and q['native']['all8normal_before_communicators']
  for z in q['native']['reports']:
   assert z['global_tp_rank']==node*8+z['rank']and z['actual_tp_size']==16 and z['normal_Runtime_close']and len(z['arms'])==10;rows.append(z)
   for i,a in enumerate(z['arms']):
    assert a['candidate_ON']==bool(plan[i]);m=a['markers'];on=6<=i<=9
    if not on:assert m is None;continue
    assert m['device']==z['rank']and m['marker_count']==16 and m['Runtime_identity']>0
    intervals=m['intervals'];assert len(intervals)==15 and [intervals[0]['from']]+[p['to']for p in intervals]==names
    h=m['routing_histogram'];assert h['original_routes']==256 and h['original_even_routes']+h['original_odd_routes']==256 and h['effective_routes']==h['masked_valid_routes']==sum(h['effective_routes_per_token']) and h['actual_jobs_TT2']==h['expected_jobs_TT2']
    assert all(math.isfinite(p[k])and p[k]>=0 for p in intervals for k in ['stream_span_ms','host_wall_ms','threadCPU_ms']);assert abs(sum(p['stream_span_ms']for p in intervals)-m['whole_forward_stream_span_ms'])<.02
 assert len(rows)==16;out={'all16_schema_owner_complete':True,'arm_full_forward':[],'phases':[],'scope':'Same-owner stream intervals include hostenqueue/dependency/TPwait, not activeGPU. Host/threadCPU/stream overlap, neveradd. Per-phase maxima notsummed acrossranks; nohostclocksubtraction.'}
 for arm in range(2,10):
  v=[z['arms'][arm]for z in rows];out['arm_full_forward'].append({'arm':arm,'EP2':bool(plan[arm]),'markers_ON':6<=arm<=9,'max_owner_forward_ms':max(a['full_MoE_complete_wall_ms']for a in v),'owner_mean_forward_ms':statistics.mean(a['full_MoE_complete_wall_ms']for a in v),'max_owner_enqueue_ms':max(a['enqueue_host_ms']for a in v),'max_owner_final_drain_ms':max(a['final_drain_host_ms']for a in v),'owner_mean_enqueue_threadCPU_ms':statistics.mean(a['enqueue_threadCPU_ms']for a in v),'owner_mean_drain_threadCPU_ms':statistics.mean(a['final_drain_threadCPU_ms']for a in v)})
  if not 6<=arm<=9:continue
  for idx in range(15):
   phase=[a['markers']['intervals'][idx]for a in v];out['phases'].append({'arm':arm,'EP2':bool(plan[arm]),'from':names[idx],'to':names[idx+1],'owner_stream_span_ms':{'min':min(p['stream_span_ms']for p in phase),'median':statistics.median(p['stream_span_ms']for p in phase),'max':max(p['stream_span_ms']for p in phase)},'owner_host_wall_ms':{'min':min(p['host_wall_ms']for p in phase),'median':statistics.median(p['host_wall_ms']for p in phase),'max':max(p['host_wall_ms']for p in phase)},'owner_threadCPU_ms_median':statistics.median(p['threadCPU_ms']for p in phase)})
 anchors=[x for x in out['arm_full_forward']if x['arm']in[2,5]];marked=[x for x in out['arm_full_forward']if x['arm']in[6,9]];out['instrumentation_perturbation']={'OFF_anchors_max_owner_ms':[x['max_owner_forward_ms']for x in anchors],'OFF_marked_max_owner_ms':[x['max_owner_forward_ms']for x in marked],'anchor_endpoint_change_percent':100*(anchors[1]['max_owner_forward_ms']/anchors[0]['max_owner_forward_ms']-1),'marked_mean_vs_anchor_mean_percent':100*(statistics.mean(x['max_owner_forward_ms']for x in marked)/statistics.mean(x['max_owner_forward_ms']for x in anchors)-1),'limit':'Markersadd11eventrecords+currentdevice/capture checks; perturbation/comparable drift caninvalidate attribution. Anchorsneverincludedphase sums; noordinaryperformancegainclaim.'};out['routing_histograms']=[{'globalrank':z['global_tp_rank'],'arms':[{'arm':i,'EP2':bool(plan[i]),'hist':z['arms'][i]['markers']['routing_histogram']}for i in range(6,10)]}for z in rows];return out
if __name__=='__main__':
 handle=sys.argv[1];assert handle.startswith('910374.')and handle.split('.')[1].isdigit();paths=[r/'results'/f'Kimi-cyclic-EP2-r8669zp-{handle}-node{node}.json'for node in range(2)];out=analyze([json.loads(p.read_text())for p in paths]);out.update(actual_handle=handle,report_SHA256=[hashlib.sha256(p.read_bytes()).hexdigest()for p in paths]);p=r/'results'/f'Kimi-provider-phases-summary-r8669zn-{handle}.json';assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps({'summary':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'all16_schema_owner_complete':True}))
