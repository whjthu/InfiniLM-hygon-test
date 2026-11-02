from pathlib import Path
import collections,datetime,hashlib,json,math,statistics,tarfile
import numpy as np
import argparse
parser=argparse.ArgumentParser(description='Recompute immutable Kimi391 scalars and BF16 warm outputs without Runtime/GPU');parser.add_argument('--archive');parser.add_argument('--output');args=parser.parse_args()
G=Path(__file__).resolve().parents[1];L=G.parent
sha=lambda b:hashlib.sha256(b).hexdigest()
ref=json.loads((G/'results/bw1000/2026-11-02/Kimi391-726-original-evidence-sealed-reference.json').read_bytes())
archive=Path(args.archive)if args.archive else L/'Kimi391-original-evidence-sealed.tar.gz';assert sha(archive.read_bytes())==ref['archive_SHA256']=='35a3f742151c368fb3b6abc7f091acca2bc9cec042387c52cf4e11cfc318d284'
expected={e['name']:e for e in ref['entries']};data={}
with tarfile.open(archive,'r:gz')as t:
    for m in t.getmembers():
        if m.isdir():continue
        assert m.isfile();name=Path(m.name).name;assert name in expected and name not in data
        b=t.extractfile(m).read();e=expected[name];assert len(b)==e['nbytes'] and sha(b)==e['SHA256'];data[name]=b
assert set(data)==set(expected)
w=json.loads(data['Kimi391-GPU-watch-v1.json']);assert w['completed'] and w['passed'] and w['normal16'] and w['srun_exit0'] and w['Slurm_COMPLETED0']
assert w['actual_step']=='910371.726' and w['GPU_source_manifest_SHA256']=='cea8b0f6832af694ab5188576fbdd9901843e652534132f8260e4fd1ce2bdce1'
assert all(x['passed'] and not x['KFD_pids'] and not x['expected_PIDs_still_exist'] and not x['listeners'] and len(x['bindable_ports'])==16 for x in w['post'])
account=data['actual-accounting-original-reader.txt'].decode().strip().split('|');assert account[0]=='910371.726' and account[2:4]==['COMPLETED','0:0']
mf=json.loads(data['GPU_SOURCE_MANIFEST.json']);assert sha(data['GPU_SOURCE_MANIFEST.json'])==w['GPU_source_manifest_SHA256']
ranks={};nodes=[]
for node in range(2):
    name=f'Kimi391-910371.726-node{node}.json';q=json.loads(data[name]);assert q['completed'] and q['normal_CUDA_cleanup'] and q['Torch_after_Runtime_close']
    assert q['library_SHA256']==mf['library_SHA256'] and q['Native_SHA256']==mf['Native_SHA256']
    qref=next(x for x in w['owners']if x['node']==node);assert qref['SHA256']==sha(data[name])
    nodes.append({k:q[k]for k in ['node','weight_load_seconds','CPU_weight_bytes','inputs','providers']})
    for r in q['native']['reports']:
        n=r['global_rank'];assert n not in ranks and r['completed'] and r['normal_Runtime_close'] and r['module_and_observer_closed'] and len(r['trials'])==102
        for t in r['trials']:
            s=t['strategy'];assert t['counters_pass'] and t['input_readonly'] and t['output_tail_canary'] and t['error_vs_A']['screen_pass'] and t['ownrepeat']['screen_pass']
            assert t['I192_counts_delta']==([8,512]if s==2 else [0,0])
            assert t['helper_metadata_counts_delta'][:2]==([1,512]if s==0 else [0,0])
            assert t['I192_T_hist_delta']==[8 if s==2 and i==64 else 0 for i in range(65)]
        ranks[n]=r
assert set(ranks)==set(range(16)) and nodes[0]['inputs']==nodes[1]['inputs']
key=lambda t:(t['fixture'],t['block'],t['phase'],t['rep'])
rr={n:{key(t):t for t in r['trials']if not t['warm']}for n,r in ranks.items()};assert all(len(v)==96 and set(v)==set(rr[0])for v in rr.values())
aligned=[]
for k in sorted(rr[0]):
    s=rr[0][k]['strategy'];assert all(v[k]['strategy']==s for v in rr.values())
    times=[v[k]['full_strategy_wall_ms']for v in rr.values()];assert all(math.isfinite(x) and x>0 for x in times)
    aligned.append(dict(fixture=k[0],block=k[1],phase=k[2],rep=k[3],strategy=s,max16_ms=max(times)))
summaries=[];original=json.loads(data['Kimi391-terminal-audit-v1.json'])
for fixture in range(2):
    blocks=[]
    for block in range(4):
        means={s:statistics.fmean(t['max16_ms']for t in aligned if t['fixture']==fixture and t['block']==block and t['strategy']==s)for s in range(3)}
        blocks.append(dict(block=block,A_ms=means[0],B_ms=means[1],C_ms=means[2],C_over_A_rate_gain=means[0]/means[2]-1,C_over_B_rate_gain=means[1]/means[2]-1))
    gains=[b['C_over_A_rate_gain']for b in blocks];drift=abs(blocks[-1]['A_ms']/blocks[0]['A_ms']-1);gate=mf['performance_gate']
    keep=statistics.median(gains)>=gate['minimum_median_C_over_A_rate_gain_per_fixture'] and sum(g>0 for g in gains)>=gate['minimum_positive_blocks_per_fixture'] and drift<=gate['maximum_A_endpoint_drift']
    assert abs(statistics.median(gains)-original['summary'][fixture]['median_C_over_A_rate_gain'])<1e-12
    summaries.append(dict(fixture=fixture,blocks=blocks,means_ms={str(s):statistics.fmean(b[['A_ms','B_ms','C_ms'][s]]for b in blocks)for s in range(3)},median_C_over_A_rate_gain=statistics.median(gains),A_endpoint_drift=drift,retained=keep))
warm=[]
for fixture in range(2):
    arrays=[]
    for s in range(3):
        b=np.frombuffer(data[f'rank0-f{fixture}-strategy{s}-warm.bin'],dtype='<u2');assert b.size==512*7168
        x=(b.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(x).all();arrays.append(x.astype(np.float64))
    for s in [1,2]:
        diff=arrays[s]-arrays[0];nr=float(np.linalg.norm(diff)/np.linalg.norm(arrays[0]));mx=float(np.max(np.abs(diff))/np.max(np.abs(arrays[0])))
        assert nr<=1/64 and mx<=1/8
        warm.append(dict(fixture=fixture,strategy=s,relative_RMS=nr,maxabs_over_Amax=mx,B_C_bytes_identical=data[f'rank0-f{fixture}-strategy1-warm.bin']==data[f'rank0-f{fixture}-strategy2-warm.bin'],scope='IndependentnumpyBF16decoder diagnostic, not forcedbitexactgate'))
assert not any(s['retained']for s in summaries)
result=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='ROOT_ORIGINAL_BYTES_RECOMPUTED_NUMERIC_PASS_STRATEGY_PERFORMANCE_FAIL_STOP_CLEAN',actual_GPU='910371.726 COMPLETED0/479s',post='910371.727 two nodes PID/KFDempty/noLISTEN/allportsbindable',source_publication='1d5b3a0d30fdbbd571feba4788555067a3378fb4',terminal_publication='6268a27f981b998b87cb7ded5d3740e48047378d',GPU_manifest_SHA256=w['GPU_source_manifest_SHA256'],CPU_actual='910407.783',CPU_receipt_SHA256=mf['CPU_receipt_SHA256'],wrapper_SHA256=mf['library_SHA256'],Native_SHA256=mf['Native_SHA256'],archive=str(archive),archive_SHA256=ref['archive_SHA256'],original_entries_verified=12,independent_aligned_max16_trials=96,all16_normal=True,first_A_B_C_path_counters=True,numeric_warm_independently_recomputed=warm,per_fixture=summaries,performance_gate=mf['performance_gate'],weight_load_s=[n['weight_load_seconds']for n in nodes],CPU_weight_bytes_per_node=[n['CPU_weight_bytes']for n in nodes],residency_GPU_card_seconds=16*479,card_seconds_are_not_busy=True,actual_PCI_peak_or_HBM_traffic='UNMEASURED',Model_gain=False,real_prefill_activation=False,scope='One currentrealcheckpointMoElayerTP16, twofixedteacher16row expansions to512 geometry. FullMoE8x includes shared/projection/collectives/stitch/temp destruction/drain. No Model/KDA/MLA/realprefill/servicequalification.',decision='STOP512microchunk strategy: currentproduction512helper ~6ms versus8x64I192 ~50ms. Retain C/B same64dispatchlocalbenefit only. No grid/Modelreload/promotion or application of these ratios towholeModel.',new_GPU_or_jobs_by_root=0,review167_after_actual_causal_round=1)
out=Path(args.output)if args.output else G/'results/bw1000/2026-11-02/Kimi391-root-audited-full-MoE-strategy-negative-normal-clean.json'
with out.open('x',encoding='utf-8',newline='\n')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'result':str(out),'SHA256':sha(out.read_bytes()),'summaries':summaries,'warm':warm},ensure_ascii=False))
