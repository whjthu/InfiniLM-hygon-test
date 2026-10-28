"""Read actual persisted raw owner events; missing markers are not estimates."""
import argparse,json,statistics
from pathlib import Path
NAMES=['MoE.router.begin','MoE.router.end','MoE.down.end','MoE.provider.entry','MoE.clear.end','MoE.pack_routes.end','MoE.build_jobs.end','MoE.fused_W13_dequant_Situ.end','MoE.fused_W2_weighted_scatter.end','MoE.provider.return','MoE.routed_TP_collective.end','MoE.routed_norm.end','MoE.up.end','MoE.shared_including_TP_reduce.end','MoE.add.end']
PHASES=['router','down_projection','provider_host_gap','clear_counts_jobs','pack_routes','build_jobs','fused_W13_dequant_Situ','fused_W2_weighted_scatter','provider_return_gap','routed_TP_collective_with_arrival_wait','routed_norm','up_projection','shared_including_TP_reduce','add']
def analyze(reports):
 windows=[]
 for report in reports:
  for profile in report.get('sparse_model_profiles',[]):
   raw=profile.get('raw_decoder_owner_events')
   if raw is None:continue
   assert raw['all8_published']
   for owner in raw['owners'].values():
    for call in owner['calls']:
     for layer in [1,47]:
      points=[p for p in call['points']if p['layer']==layer and p['name'].startswith('MoE.')]
      if not points:continue
      assert [p['name']for p in points]==NAMES,('incomplete actual phase coverage',layer)
      assert all(p['from']==NAMES[i-1]and p['from_layer']==layer for i,p in enumerate(points)if i>0)
      windows.append({'label':profile['label'],'node':report['node'],'device':owner['device'],'global_TP_rank':owner['global_TP_rank'],'PP_stage':owner['PP_stage'],'layer':layer,'forward_ordinal':call['forward_ordinal'],'phase_ms':{name:points[i+1]['stream_span_ms']for i,name in enumerate(PHASES)},'original_tail_drain_ms':call['original_tail_drain_ms']})
 assert windows,'No actual phase evidence; no performance inferred'
 summary=[]
 for layer in [1,47]:
  rows=[x for x in windows if x['layer']==layer];assert rows
  summary.append({'layer':layer,'owner_windows':len(rows),'phase_ms_median':{p:statistics.median(x['phase_ms'][p]for x in rows)for p in PHASES},'phase_ms_max':{p:max(x['phase_ms'][p]for x in rows)for p in PHASES}})
 return {'actual_phase_evidence':True,'workload':'original136b real GSM32/C8W8 selectedT64 verify afterwarm','histogram':'DEFERRED','pure_CCL_or_GPU_busy':False,'crosshost_clock_subtraction':False,'do_not_add_PPstage_whole_times':True,'summary':summary,'windows':windows}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('reports',nargs=4);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);assert not out.exists();q=analyze([json.loads(Path(s).read_bytes())for s in a.reports]);out.write_text(json.dumps(q,indent=2));print(json.dumps({'actual_phase_evidence':True,'windows':len(q['windows']),'histogram':'DEFERRED'}))
