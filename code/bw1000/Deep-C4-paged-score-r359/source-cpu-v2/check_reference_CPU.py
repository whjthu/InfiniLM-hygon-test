"""Independent scalar F32/tree checks for the vector CPU score oracle only."""
import json
import math
import struct
import numpy as np
from reference import decode_fp8, score_cpu, gather_paged, geometry


def f32(value):
    return struct.unpack('<f', struct.pack('<f', float(value)))[0]


def scalar(query, weights, keys, scales):
    output = []
    for key, scale in zip(keys, scales):
        contribution = [0.]*128
        for head in range(8):
            partial = []
            for lane in range(16):
                acc = 0.
                for d in range(lane, 128, 16):
                    acc = f32(acc + f32(decode_fp8(int(query[head, d]))*decode_fp8(int(key[d]))))
                partial.append(acc)
            for span in [8, 4, 2, 1]:
                partial[:span] = [f32(partial[i]+partial[i+span]) for i in range(span)]
            positive = max(f32(partial[0]*float(scale)), 0.)
            contribution[head] = f32(positive*float(weights[head]))
        for span in [64, 32, 16, 8, 4, 2, 1]:
            contribution[:span] = [f32(contribution[i]+contribution[i+span]) for i in range(span)]
        output.append(contribution[0])
    return np.array(output, dtype=np.float32)


def check():
    valid = [i for i in range(256) if (i&127) != 127]
    assert len(valid) == 254
    assert decode_fp8(0) == 0 and math.copysign(1, decode_fp8(128)) == -1
    assert decode_fp8(56) == 1 and decode_fp8(8) == 2**-6
    assert decode_fp8(1) == 2**-9 and decode_fp8(126) == 448 and decode_fp8(254) == -448
    for code in valid:
        assert math.isfinite(decode_fp8(code))
        assert decode_fp8(code) == -decode_fp8(code ^ 128)
    for code in [127, 255, -1, 256, True]:
        try:
            decode_fp8(code)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid FP8 code accepted')
    rng = np.random.default_rng(359)
    query = rng.choice(valid, (8, 128)).astype(np.uint8)
    keys = rng.choice(valid, (5, 128)).astype(np.uint8)
    scales = np.array([.125, 1, 2, .25, .5], dtype=np.float32)
    weights = np.array([1, -.5, .25, 2, 0, -.125, .5, 1.5], dtype=np.float32)
    expected = scalar(query, weights, keys, scales)
    actual = score_cpu(query, weights, keys, scales, 5)
    assert np.array_equal(expected.view(np.uint32), actual.view(np.uint32))
    masked = score_cpu(query, weights, keys, scales, 3)
    assert np.array_equal(masked[:3], expected[:3]) and np.isneginf(masked[3:]).all()
    raw = np.full((128, 8448), 0xa5, dtype=np.uint8)
    slots = [0, 63, 64, 2123, 2143]
    written = [False]*8192
    for i, slot in enumerate(slots):
        page, offset = divmod(slot, 64)
        raw[page, offset*128:(offset+1)*128] = keys[i]
        raw[page, 8192+offset*4:8196+offset*4] = scales[i:i+1].view(np.uint8)
        written[slot] = True
    recovered, factors = gather_paged(raw, slots, written)
    assert np.array_equal(recovered, keys) and np.array_equal(factors[:, 0], scales)
    try:
        gather_paged(raw, [65], written)
    except AssertionError:
        pass
    else:
        raise AssertionError('Unwritten cache slot read')
    for mutation in ['query127', 'key255', 'scale0', 'scaleNaN', 'weightInf', 'visibleNegative', 'visibleTooLarge']:
        q, k, s, w = query.copy(), keys.copy(), scales.copy(), weights.copy()
        v = 5
        if mutation == 'query127': q[0, 0] = 127
        if mutation == 'key255': k[0, 0] = 255
        if mutation == 'scale0': s[0] = 0
        if mutation == 'scaleNaN': s[0] = np.nan
        if mutation == 'weightInf': w[0] = np.inf
        if mutation == 'visibleNegative': v = -1
        if mutation == 'visibleTooLarge': v = 6
        try:
            score_cpu(q, w, k, s, v)
        except ValueError:
            pass
        else:
            raise AssertionError(mutation)
    bad = query.copy(); bad[0, 0] = 127
    assert np.isneginf(score_cpu(bad, weights, keys, scales, 0)).all()
    assert geometry(8496)['after'] == 2124 and geometry(8575)['after'] == 2144
    return dict(passed=True, GPU=0, Native_dlopen=False, scope='CPU oracle only, no SDK baseline or HIP kernel executed',
                all256_codes_classified=True, valid_codes=254, invalid_codes=[127, 255],
                F32_lane_and_head_tree_bitwise_scalar_check=True, visibility_and_invalid_features=True,
                page64_scale_offset8192_checked=True, pending_boundary_geometry=True)


if __name__ == '__main__':
    print(json.dumps(check()))
