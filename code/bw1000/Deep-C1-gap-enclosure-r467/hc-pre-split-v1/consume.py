"""Existing owner/inner completion first, then new HC sibling packet, exact ABI."""
import ctypes as C,json,math
LABELS=['cast_square_mean_normalize','mixes_GEMM','affine_sigmoid_softmax','sinkhorn','weighted_collapse_cast']
def bind(lib):
 f=lib.deep_hc_pre_split_read_r467;f.argtypes=[];f.restype=C.c_char_p;return f

def consume(raw,owner):
 q=json.loads(raw);assert q['schema']==1 and q['profile']=='HCpre-r467'and q['labels']==LABELS and len(q['rows'])==688
 parents={(r['device'],r['layer']):r for r in json.loads(owner)['rows']};assert len(parents)==344;seen=set();out={k:[[0.]*8 for _ in range(3)]for k in LABELS}
 for r in q['rows']:
  key=r['device'],r['layer'],r['HC'];assert key not in seen and key[0]in range(8)and key[1]in range(43)and key[2]in[0,1];seen.add(key);p=parents[key[:2]];assert(r['owner'],r['stream'])==(p['owner'],p['stream'])and r['thread']and r['created']==r['queried']==r['destroyed']==6
  assert math.isfinite(r['original_drain_host_ms'])and r['original_drain_host_ms']>=0 and len(r['spans'])==5
  for i,s in enumerate(r['spans']):
   assert s['stage']==i
   for j,k in enumerate(['stream_ms','host_wall_ms','thread_cpu_ms']):assert math.isfinite(s[k])and s[k]>=0;out[LABELS[i]][j][r['device']]+=s[k]
  assert abs(sum(s['stream_ms']for s in r['spans'])-r['whole_stream_ms'])<.02
 assert len(seen)==688
 return dict(rows=688,HCinstances_per_owner=86,events_created_destroyed_per_window=4128,perstage_perowner_ms={k:dict(stream=v[0],host=v[1],threadCPU=v[2])for k,v in out.items()},scope='Only five siblingHCpre intervals may sum; do not add originalHCpre parent orFFNparent; threadCPU may spin; streamelapsed not pureGPUbusy.')
