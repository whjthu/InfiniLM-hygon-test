"""Recompute existing loading scalars; no jobs, producer imports or live reads."""
from pathlib import Path
import argparse,hashlib,json,statistics,tarfile
H=lambda b:hashlib.sha256(b).hexdigest()
def read(archive,current):
 raw=Path(archive).read_bytes();assert H(raw)=='78206dc20170c0afcdf59d0b05d77eaadca2c28da021466d6c7a2e9732a6c27b'
 with tarfile.open(archive)as t:
  nodes=[json.load(t.extractfile('results/Kimi-r461-full-model-tp32-full-cohort-20261105-c/node%d.json'%i))for i in range(4)]
 load=nodes[2]['full_loader'];flush=load['native_load_flush_profiles'];phase=load['host_loader_phase_profile'];total=load['loading_seconds'];assert len(flush)==741 and load['loaded_keys']==497052
 native=sum(v['native_load_host_inclusive_s']for v in flush);copy=[sum(v['r2515_delta'][i]for v in flush)for i in range(5)]
 assert sum(v['keys']for v in flush)==497052 and copy==[1320400,4737646592,126780702720,0,0]
 def qs(values):return dict(min=min(values),median=statistics.median(values),max=max(values))
 progress=[]
 for n,v in enumerate(nodes):
  rows=[e['payload']for e in v['events']if e['event']=='load_progress'];segments=[];before=dict(loaded_keys=0,loading_elapsed_seconds=0.)
  for row in rows:
   dk=row['loaded_keys']-before['loaded_keys'];dt=row['loading_elapsed_seconds']-before['loading_elapsed_seconds'];assert dk>0 and dt>0
   segments.append(dict(keys=dk,seconds=dt,keys_per_s=dk/dt));before=row
  progress.append(dict(node=n,recorded_progress=len(rows),last=rows[-1],segment_seconds=qs([s['seconds']for s in segments]),segment_keys_per_s=qs([s['keys_per_s']for s in segments]),fullprofile_available=n==2))
 currentraw=Path(current).read_bytes();now=json.loads(currentraw);v=now['Kimi643'];assert v['watch']['actual_GPU']=='910380.643'
 return dict(schema=1,existing_artifact_audit_PASS=True,new_jobs=0,new_GPU=0,Model_rerun=False,loader_SHA256='d3097bfb15750ceac90fef0994e40fafbd1e12a329356fef16f01a4853f5b5ff',original637=dict(node=2,loading_s=total,phases=phase,phase_fraction={k:p['seconds']/total for k,p in phase.items()},unattributed_host_s=total-sum(p['seconds']for p in phase.values()),flush_count=len(flush),keys=load['loaded_keys'],CPU_logical_source_bytes=sum(p['CPU_logical_source_bytes']for p in flush),native_load_inclusive_s=native,flush_native_s=qs([v['native_load_host_inclusive_s']for v in flush]),r2515_counters=copy,bytes_per_counted_memmove_block=copy[2]/copy[1],progress=progress),current643_original_snapshot=dict(UTC=now['UTC'],SHA256=H(currentraw),nodes=v['nodes'],limit='Progress snapshots only, no current perphase profiles until loader returns. Not GPUbusy/fullfit/Ready.'),measurement_limits=['native_load includes source faults/prepare/8owner loading/sync; not H2D-only','get_tensor is provider-inclusive and may include mmap/faults; not pure Python or disk-only','CPU logical descriptor bytes, memmove counted bytes and physical storage reads are different domains','Only node2 of637 completed loader; four-node current643 still loading. No cross-run cold/warm paired gain.'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('current');a=p.parse_args();print(json.dumps(read(a.archive,a.current),indent=2,sort_keys=True))
