"""Independently consume one completed CPU audit; never load captured tensors again."""
import hashlib
import json
from pathlib import Path

A=Path('/home/whj/nvidia-baseline/alignment-assets')
D=A/'hygon-llava-A100-capture-value-reader-source-v1'
O=A/'hygon-llava-A100-capture-value-reader-v1'
R=A/'hygon-llava-A100-first-updates-v1'
EXPECTED_OUTCOME='b3f30bdb750b064a0b7849cfe75ad8e4b4486ed6eade0d728b1b628756e31b73'
EXPECTED_MF='fedfc938441674d3ad73c633c65ba39c46382b704f99aa8617409c15a1e9ec02'
PUB='c865a1c57b485c35e92225c7ea55cf42bc1a03e2'
REQUEST='6624046555964d0cbd3ab52f955708e2'
H=lambda b:hashlib.sha256(b).hexdigest()
pins={}
def read(path,expected=None,cap=2*1024**2):
    assert path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=cap
    b=path.read_bytes()
    if expected:assert H(b)==expected
    pins[str(path.relative_to(A))]=dict(bytes=len(b),SHA256=H(b))
    return json.loads(b)
def main():
    mf=read(D/'SOURCE_FILES.json',EXPECTED_MF)
    for name,pin in mf['files'].items():
        p=D/name
        assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes']
        assert H(p.read_bytes())==pin['SHA256']
    binding=read(D/'LLAVA_NATIVE_BINDING.json')
    originalpins=read(D/'ORIGINAL_PINS.json')
    intent=read(O/'intent.json')
    out=read(O/'outcome.json',EXPECTED_OUTCOME)
    assert out['publication']==PUB and out['request']==REQUEST
    assert intent['publication']==PUB and intent['request']==REQUEST and intent['source_manifest_SHA256']==EXPECTED_MF
    assert (intent['whole_seconds'],intent['work_seconds'],intent['maximum_active_workers'])==(900,780,1)
    assert intent['llava_outcome_SHA256']=='cc46b557872c86fef13a04a2deaccd402268dba30d0abcf7b669a42d1162e298'
    assert intent['prior_reader_outcome_SHA256']=='a4f22c3a342f47a1c8f0dd7979f14d44fb3bd7bcd4b93197f233c89763a1e550'
    assert out['passed'] is True and out['failure'] is None
    assert (out['attempted_ranks'],out['completed_ranks'],out['completed_files'])==(8,8,200)
    assert (out['rank_denominator'],out['file_denominator'],out['update_denominator'],out['micro_forward_denominator'])==(8,200,16,64)
    assert out['actual_GPU_cards']==out['artifact_bytes_exported']==0
    assert 0<out['elapsed_seconds']<780 and 0<out['observed_worker_peak_RSS_bytes']<=8*1024**3
    assert out['tensor_finite_independently_checked'] is True and out['full_200_SHA_checked'] is True
    assert all(out[k] is False for k in ('full_optimizer_recompute','Hygon_alignment','formal_performance','formal_quality','replay_allowed','hard_memory_cgroup_enforced'))
    assert [w['rank'] for w in out['workers']]==list(range(8))
    rows=[];verifiedbytes=0
    for rank,w in enumerate(out['workers']):
        assert w['returncode']==0 and w['forced'] is None and w['stderr_bytes']==0
        assert w['stderr_SHA256']==H(b'')
        err=O/('rank-%02d.stderr'%rank)
        assert err.is_file() and not err.is_symlink() and err.read_bytes()==b''
        v=read(O/('rank-%02d.json'%rank),w['stdout_SHA256'])
        assert v==w['value'] and v['rank']==rank and v['passed'] is True
        assert (v['files_verified'],v['files_denominator'],v['updates_denominator'],v['micro_forwards_denominator'])==(25,25,2,8)
        assert v['initial_parameter_SHA_matches_native_rank'] is True and v['independent_tensor_finite'] is True
        assert v['actual_GPU_cards']==0 and v['cpu_only'] is True and v['full_optimizer_recompute'] is False
        assert v['torch_version']=='2.8.0a0+5228986c39.nv25.06'
        rel='capture/rank-%02d/receipt.json'%rank
        receipt=read(R/rel,originalpins['originals'][rel]['SHA256'])
        assert receipt['rank']==rank and len(receipt['files'])==25
        fs={f['name']:f for f in receipt['files']}
        assert len(fs)==25 and len(v['files'])==25 and {f['name'] for f in v['files']}==set(fs)
        expectednames={'initial.pt'}
        for micro in range(1,9):
            expectednames.update('forward-%02d-%s.pt'%(micro,kind) for kind in ('input','output'))
        for update in (1,2):
            expectednames.update('update-%02d-%s.pt'%(update,kind) for kind in ('preclip','postclip','before-optimizer','after-optimizer'))
        assert set(fs)==expectednames
        for f in v['files']:
            original=fs[f['name']]
            assert f['bytes']==original['bytes'] and f['SHA256']==original['SHA256']
            assert f['independent_values_finite'] is True and f['stable_same_fd_SHA_load'] is True
            assert f['tensor_count']>0 and f['tensor_elements']>0
            digest=f['shape_dtype_inventory_SHA256']
            assert len(digest)==64 and all(c in '0123456789abcdef' for c in digest)
        count=sum(f['bytes'] for f in fs.values())
        assert count==v['capture_bytes']==originalpins['capture_bytes_per_rank'][rank]
        verifiedbytes+=count
        assert [u['update'] for u in v['updates']]==[1,2]
        for u in v['updates']:
            assert u['parameters_denominator']==504 and u['gradient_parameters']+u['none_gradients']==504
            assert 0<=u['parameters_changed']<=504 and u['clip_numerical_parity_qualified'] is False
            for side in ('optimizer_before','optimizer_after'):
                op=u[side]
                assert op['initialized']+op['empty']==504
                assert op['step']==u['update']-(side=='optimizer_before')
                assert len(op['state_values_SHA256'])==64
        assert v['updates'][0]['optimizer_after']['state_values_SHA256']==v['updates'][1]['optimizer_before']['state_values_SHA256']
        rows.append(dict(rank=rank,files=25,capture_bytes=count,updates=v['updates']))
    assert verifiedbytes==34774313192
    result=dict(state='ORIGINAL_CPU_VALUE_AUDIT_INDEPENDENT_TERMINAL_PASS',audit_request=REQUEST,publication=PUB,original_training_request='640a2e0f2ae4440a9689912f2de413be',audit_outcome_SHA256=EXPECTED_OUTCOME,source_manifest_SHA256=EXPECTED_MF,
        completed_ranks=8,rank_denominator=8,completed_files=200,file_denominator=200,optimizer_updates=16,micro_forwards=64,
        logical_capture_SHA_stream_bytes=verifiedbytes,elapsed_seconds=out['elapsed_seconds'],observed_worker_peak_RSS_bytes=out['observed_worker_peak_RSS_bytes'],
        sampled_RSS_not_cgroup=True,workers_sequential=True,CPU_affinity_cores=2,CPU_threads=2,actual_GPU_cards=0,artifact_bytes_exported=0,
        full_artifact_SHA_and_finite=True,initial_parameter_and_optimizer_boundary_continuity=True,clip_numerical_parity_qualified=False,
        full_optimizer_recompute=False,Hygon_alignment=False,formal_performance=False,formal_quality=False,checkpoints_10_20_25_content_qualified=False,
        ranks=rows,metadata_originals=pins,capture_tensor_reloads_by_terminal_reader=0,replay_allowed=False)
    print(json.dumps(result,allow_nan=False))
if __name__=='__main__':main()
