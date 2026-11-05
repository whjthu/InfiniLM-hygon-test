"""Source-level Kimi tiling cost, not a performance or fit prediction."""
import argparse,hashlib,json,math
from pathlib import Path
CASES=[(512,0),(512,3071),(512,32255),(1,32767),(512,0)]

def count(q,past,Q,K):
    first=calls=packed=tiles=0
    while first<q:
        rows=min(Q,q-first,K-(past+first)%K)
        end=past+first+rows
        calls+=math.ceil(end/K)
        packed+=end*3*192*2*2
        first+=rows;tiles+=1
    return dict(FA_calls=calls,Q_tiles=tiles,packed_logical_bytes=packed)

def cost(Q,K):
    require=1<=Q<=32 and K>0 and K%256==0
    if not require:raise ValueError('candidate outside declared source study')
    a=[count(q,p,Q,K) for q,p in CASES]
    # kt,vt BF16; partial BF16; lse/state/factors/sum FP32 from actual source.
    staging=2*K*3*192*2 + Q*3*192*2 + Q*3*4 + 2*Q*3*2*4 + Q*3*128*4 + 4
    return dict(Q=Q,K=K,per_arm_cases=a,ABBA_FA_calls=4*sum(v['FA_calls'] for v in a),
                explicit_staging_bytes=staging,source_estimate_only=True,
                SDK_workspace_bytes=None,allocator_peak_bytes=None,Model_fit=False,performance=None)

def study():
    baseline=cost(32,3072);rows=[]
    for Q in [8,16,32]:
        for K in [1536,3072,6144,12288]:
            r=cost(Q,K);r['extra_explicit_staging_bytes']=r['explicit_staging_bytes']-baseline['explicit_staging_bytes']
            r['within_48MiB_component_surplus_estimate']=r['extra_explicit_staging_bytes']<=48*1024**2
            r['next']=('REUSE_664_BASELINE' if (Q,K)==(32,3072) else 'SOURCE_SUPPORTS_AND_NUMERIC_REQUIRED')
            rows.append(r)
    # A weakly dominated source estimate has no justification for the first GPU
    # wave. No measured SDK workspace/performance relationship is asserted.
    for r in rows:
        r['static_dominated']=any(s['ABBA_FA_calls']<=r['ABBA_FA_calls'] and s['explicit_staging_bytes']<=r['explicit_staging_bytes'] and
            (s['ABBA_FA_calls']<r['ABBA_FA_calls'] or s['explicit_staging_bytes']<r['explicit_staging_bytes']) for s in rows)
    return dict(schema='Kimi-tiling-static-cost-v1',baseline=baseline,candidates=rows,
                preserved='full32768/logical512/H3/K192/V128/causal-boundary/F32LSE merge/4GiB',
                source_changes_required=['contract.hpp tile constants','fa_backend.cpp buffer dimensions and chunkstep','fa_kernels.cu launch limits'],
                existing_664_repeated=False,GPU_runs=0,quality_qualified=False)

if __name__=='__main__':print(json.dumps(study(),indent=2,allow_nan=False))
