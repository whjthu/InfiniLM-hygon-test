"""Read-only actual mixed-cache identity gate, separate from HTTP/gold scoring.

No Native/weights/job imports. A future deployment must call this with the
original16 reports/censuses/ready rows; it does not qualify unexecuted output.
"""
from pathlib import Path
import argparse,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'provider'))
from public_glm_native.cache_policy_r452 import global_metadata,for_config,validate_census
NATIVE='9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'
def validate_originals(config,source_sha,owners,constructors,records,ready,*,require_normal=True):
    if len(owners)!=16 or len(constructors)!=16 or len(records)!=16:raise ValueError('Exactly16 original owners/censuses/ready records')
    global_policy=global_metadata(config)
    if ready['cache_policy']!=global_policy or ready['Native_SHA256']!=NATIVE:raise ValueError('MODEL_READY/cache/Native mismatch')
    signature=[]
    for stage in range(2):
        p=for_config(config,stage);same=[]
        for rank in range(8):
            i=stage*8+rank;q=owners[i];c=constructors[i];r=records[i]
            if q['cache_policy']!=p.metadata()or c['cache_policy']!=p.metadata()or r['cache_policy']!=p.metadata():raise ValueError('TP owner/stage policy mismatch')
            if (c['stage'],c['rank'],r['stage'],r['rank'])!=(stage,rank,stage,rank):raise ValueError('Original stage/rank identity mismatch')
            if q['source_manifest_SHA256']!=source_sha or c['source_manifest_SHA256']!=source_sha or q['library_sha256']!=NATIVE or c['Native_SHA256']!=NATIVE or r['Native_SHA256']!=NATIVE:raise ValueError('Original source/Native mismatch')
            validate_census(c['census'],p)
            if q['constructor_cache_census']!=c['census']or not q['constructor_cache_verified']:raise ValueError('Owner/census mismatch')
            if q['exact_bank_dispatch_r4188'][0]!=p.exact_bank_count:raise ValueError('Original exact-bank dispatch mismatch')
            if not q['all16_preweight_8GiB_passed']or not q['all16_loaded_and_postload_4GiB_passed']:raise ValueError('Unchanged8GiB/4GiB admission failed')
            if require_normal and not all(q.get(k)for k in['completed','normal_shutdown','worker_shutdown_completed','post_close_Torch_liveness']):raise ValueError('Original owner lifecycle not normal')
            if any(row['cache_policy']!=p.metadata()for row in q.get('service_requests',[])):raise ValueError('Request-time cache identity drift')
            same.append(q['cache_policy'])
        if not all(v==same[0]for v in same):raise ValueError('All8TP stage agreement')
        signature.append(p.metadata())
    return dict(passed=True,actual_owner_count=16,cache_policy=global_policy,stages=signature,Native_SHA256=NATIVE,extra_stage0_bytes_per_rank=5100273664 if global_policy['policy']=='stage0-fullV-stage1-NoV'else 0,physical_peak_or_fit_proven=False,Model_quality_or_performance_proven=False)
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--run-folder',required=True);p.add_argument('--config',required=True);p.add_argument('--source-sha',required=True);a=p.parse_args(argv);root=Path(a.run_folder);read=lambda name:json.loads((root/name).read_bytes())
    owners=[read('owner-%d-%d.json'%(s,r))for s in range(2)for r in range(8)]
    constructors=[read('constructor-%d-%d.json'%(s,r))for s in range(2)for r in range(8)]
    records=[read('ready-%d-%d.json'%(s,r))for s in range(2)for r in range(8)]
    result=validate_originals(json.loads(Path(a.config).read_bytes()),a.source_sha,owners,constructors,records,read('MODEL_READY.json'));print(json.dumps(result));return 0
if __name__=='__main__':raise SystemExit(main())
