"""Preserve complete adjacent hook tape; repeated nonstructural names remain legal."""
import math
REQUIRED=['layer.begin','decoder.RMS_attention.begin','decoder.attention.begin','attn.prepare.begin','attn.indexer.begin','attn.projectedKV.begin','attn.cache.begin','attn.row.compute.begin','attn.row.compute.end','attn.output.begin','attn.output.end','decoder.attention.end','decoder.RMS_ffn.begin','decoder.FFN.begin','FFN.router.end','FFN.gateup.begin','FFN.gateup.end','FFN.downquant.begin','FFN.down.begin','FFN.down.end','FFN.shared.begin','FFN.shared.end_TP.begin','FFN.TP.end_cast.begin','decoder.FFN.end','layer.return']
FIELDS=['stream_timeline_ms_includes_host_enqueue_idle','host_wall_ms','threadCPU_ms']
def validate_profile(report,mode):
 assert mode in [0,1]and report['N']==report['B']==1 and report['profile']==mode and report.get('layers',[report.get('layer')])==[38]
 stages=report['stages']
 if not mode:assert not stages;return
 assert stages
 for i,stage in enumerate(stages):
  if i:assert stages[i-1]['to']==stage['from']
  assert all(math.isfinite(stage[k])and stage[k]>=0 for k in FIELDS)
 names=[stages[0]['from']]+[s['to']for s in stages];assert names[0]=='layer.begin'and names[-1]=='layer.return'
 assert all(names.count(n)==1 for n in REQUIRED)
 assert[n for n in names if n in REQUIRED]==REQUIRED
