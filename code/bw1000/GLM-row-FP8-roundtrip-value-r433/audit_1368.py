"""Offline audit of the original failed component; never initializes a device.

Usage: python audit_1368.py original.json.gz [--output result.json]
NumPy is used only for F32 interpretation of actual saved device tensors.
"""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

SEAL = '905dd63e3307735a46bfbedd59a587ffa776bf499184260566a29bb898e6c2fa'
GPU_MF = 'd6ea5a9a46921c28338b5bab64742d55feeb9b3b5e59244d7479a3ac30487b9e'
LIB = '80a14a45954f4eaeed2dfca64f9be9f7ceb47d404e6e7a746a9c2062b7a285db'
ROOT = '/public/home/aclsylqidf/hygon-inference-refactor-20261014'


def audit(path):
    import numpy as np
    packed = Path(path).read_bytes()
    assert hashlib.sha256(packed).hexdigest() == SEAL
    seal = json.loads(gzip.decompress(packed))
    files = {}
    for name, record in seal['files'].items():
        data = base64.b64decode(record['bytes'], validate=True)
        assert len(data) == record['nbytes']
        assert hashlib.sha256(data).hexdigest() == record['SHA256']
        files[name] = data
    def document(suffix):
        values = [v for k, v in files.items() if k.endswith(suffix)]
        assert len(values) == 1
        return json.loads(values[0])
    watch = document('/GLM433-watch-v1.json')
    assert watch['completed'] and not watch['passed']
    assert watch['actual_step'] == '910374.1368'
    assert watch['GPU_source_manifest_SHA256'] == GPU_MF
    assert watch['owned_failure_cleanup']['closed']
    post = watch['failure_post']
    assert post['actual_step'] == '910374.1369' and post['passed']
    for key in ['KFD_pids', 'matching_GPU_driver_PIDs', 'expected_PIDs_still_exist']:
        assert post[key] == []
    assert len(post['expected_owner_PIDs']) == 9
    ranks = []
    for rank in range(8):
        r = document('/GLM433-910374.1368-rank%d.json' % rank)
        assert r['rank'] == rank and r['library_SHA256'] == LIB
        assert not r['completed'] and not r['normal']
        assert r['destroy_attempted_once'] and r['destroy_RC'] == 0 and r['Runtime_expired']
        assert len(r['records']) == 2
        off, on = r['records']
        assert off['mode'] == 0 and off['warm'] and off['Native_CAPI_completed']
        assert off['qualification_passed'] and off['raw_outputs_complete']
        assert on['mode'] == 1 and on['warm'] and on['Native_CAPI_invoked']
        assert on['Native_CAPI_returned'] and on['Native_CAPI_RC'] == 1
        assert not on['Native_CAPI_completed'] and not on['qualification_passed']
        assert on['query_error'] == "RuntimeError('requested operator implementation does not support these parameters')"
        d = on['typed_diagnostic']
        assert d['collection_passed'] and d['clear_RC'] == 0
        assert d['descriptor']['quiesced'] and not d['descriptor']['query_CAPI_completed']
        typed = {v['name']: v for v in d['files']}
        def array(name):
            v = typed[name]
            data = files[ROOT + v['path'][len('/refactor'):]]
            assert len(data) == v['nbytes'] and hashlib.sha256(data).hexdigest() == v['SHA256']
            a = np.frombuffer(data, dtype={'F32': '<f4', 'BF16': '<u2', 'U8': 'u1'}[v['dtype']]).reshape(v['shape'])
            return (a.astype(np.uint32) << 16).view(np.float32) if v['dtype'] == 'BF16' else a
        expected = np.eye(8, 512, dtype=np.float32)
        assert np.array_equal(array('identity_output_F32'), expected)
        item = dict(rank=rank, phase=d['descriptor']['phase'], identity_exact=True,
                    clear_RC=d['clear_RC'], destroy_RC=r['destroy_RC'], Runtime_expired=r['Runtime_expired'])
        if rank == 0:
            assert d['descriptor']['phase'] == 'F32rowRoundtrip'
            def decode(x):
                e = ((x >> 3) & 15).astype(np.int32)
                m = (x & 7).astype(np.float32)
                a = np.where(e == 0, np.ldexp(m / np.float32(8), -6),
                             np.ldexp(np.float32(1) + m / np.float32(8), e - 7)).astype(np.float32)
                a = np.where((x & 128) != 0, -a, a).astype(np.float32)
                return np.where((e == 15) & (m == 7), np.nan, a)
            x = decode(array('packed_latent_U8')) * np.repeat(array('row_scale_F32'), 128, axis=1)
            assert np.isfinite(x).all() and np.array_equal(x, array('row_roundtrip_F32'))
            w = decode(array('original_W_U8'))
            w *= np.repeat(array('global_W_scale_F32')[np.arange(3584) // 128], 128, axis=1)
            for h in range(8):
                assert np.array_equal(w[h * 448 + 192:h * 448 + 448].T, array('Vhead%d_F32' % h))
            p = array('P_BF16')
            assert np.isfinite(p).all() and (p >= 0).all()
            assert 'aggregate_F32' not in typed and 'attention_precast_F32' not in typed
            item.update(row_roundtrip_exact_F32=True, global128_head448_V_mapping_exact=True,
                        P_finite_nonnegative=True, diagnostic_bytes=d['descriptor']['total_bytes'])
        ranks.append(item)
    accounts = {r[0]: r[2:5] for r in seal['account_rows']}
    assert accounts['910374.1368'][:2] == ['FAILED', '1:0']
    assert accounts['910407.882'][:2] == ['FAILED', '1:0']
    assert accounts['910374.1369'][:2] == ['COMPLETED', '0:0']
    return dict(offline_audit_passed=True, component_passed=False, status='STOP_UNSUPPORTED',
                seal_SHA256=SEAL, files=len(files), source_bytes=sum(map(len, files.values())),
                actual_accounts=accounts, attempted_returned_CAPI=16, successful_CAPI=8,
                qualified_ON=0, formal_queries=0, tail_reached=False, ranks=ranks,
                failure_window=['BF16 probability to F32 Cast', 'any of 8 FullFp32 aggregation GEMMs and 8 interleaved FullFp32 value GEMMs'],
                exact_refusing_operator='UNKNOWN: no phase marker or saved per-submit counters between Cast and completion of all16 GEMMs',
                complete_candidate_numeric='UNKNOWN', candidate_performance='UNKNOWN',
                Model_quality='UNKNOWN', no_retry=True, failure_post=post)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('seal')
    p.add_argument('--output')
    args = p.parse_args()
    result = audit(args.seal)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output:
        Path(args.output).write_text(text, encoding='utf8')
    print(text)
