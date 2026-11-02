"""Offline original forty NPZ, same CPU FP8 oracle; no Native/GPU/Model/jobs."""
from pathlib import Path
import argparse,json,sys,hashlib
def main(report,taskroot):
 import numpy as np
 root=Path(taskroot);host=lambda p:root/Path(p).relative_to('/refactor')if str(p).startswith('/refactor/')else Path(p)
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 sys.path.insert(0,str(root/'incoming/Deep-C4-status-enclosure-r406/cpu-v2'));sys.path.insert(0,str(root/'incoming/Deep-C4-status-enclosure-r406/gpu-v1'))
 import reference as oracle,consumer
 q=json.loads(Path(report).read_bytes());previous=None;metrics=[]
 for item in q['plan']:
  packets=[]
  for rank in range(8):
   rec=next(r for r in q['raw_archives']if r['query']==item['index']and r['rank']==rank);jp=host(rec['metadata']);npz=host(rec['npz']);assert sha(jp)==rec['metadata_SHA256']and sha(npz)==rec['npz_SHA256'];m=json.loads(jp.read_bytes());fields={}
   with np.load(npz,allow_pickle=False)as z:
    for key,v in m['fields'].items():
     if v is None:fields[int(key)]=None;continue
     raw=z[v['array']].copy();assert hashlib.sha256(raw.tobytes()).hexdigest()==v['SHA256'];fields[int(key)]=dict(raw=raw,shape=tuple(v['shape']),dtype=v['dtype'])
   packets.append(dict(rank=rank,item=m['item'],stats=m['stats'],fields=fields))
  # Actual metadata dtype codes come from the original CABI/field roles, never guessed BF16 size.
  f=packets[0]['fields'];dtypes=dict(U8=f[0]['dtype'],F32=f[1]['dtype'],I64=f[5]['dtype'],BF16=-999)
  assert all(p['fields'][7]['dtype']==dtypes['F32']for p in packets),'This actual fixture state is F32; unknown dtype must not be silently interpreted'
  v=consumer.qualify(packets,oracle,dtypes,previous);previous=packets;metrics.append(dict(query=item['index'],timing=item['timing'],max_CPU_FP8_NRMSE=max(r['CPU_FP8_score_NRMSE']for r in v),owners=v))
 return dict(passed=True,report_SHA256=sha(report),owner_queries=40,FP8_oracle_recomputed_from_actual_NPZ=True,GPU_initialized=False,Native_dlopen=False,jobs_submitted=0,metrics=metrics)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--taskroot',required=True);a=p.parse_args();print(json.dumps(main(a.report,a.taskroot),indent=2))
