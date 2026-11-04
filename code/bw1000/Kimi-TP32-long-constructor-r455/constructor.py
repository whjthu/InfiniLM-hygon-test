"""TP32PP1 constructor-only consumer. No library loading, weights or HTTP.

The caller supplies an already verified library and metadata. This module is
not a device launcher. Current CPU961 artifact is blocked by the r455 capacity
audit; construction must not be scheduled until that audit is resolved.
"""
import ctypes as C
import json
import threading
from contextlib import contextmanager
from weight_key_guard_r9156 import summarize_keys

P, I, S, B = C.c_void_p, C.c_int, C.c_size_t, C.c_char_p
SIGNATURES = {
    'modern_kimi_engine_create_long_geometry_r453': ([B, I, I, I, B, I], P),
    'modern_kimi_long_geometry_capabilities_r453': ([P], B),
    'modern_kimi_engine_keys_r2475': ([P], B),
    'modern_kimi_engine_layout_r2475': ([P], B),
    'modern_kimi_engine_cache_geometry_r2475': ([P], B),
    'modern_kimi_engine_output_metrics_r2475': ([P], B),
    'modern_kimi_engine_partition_r2475': ([I, I, C.POINTER(S)], I),
    'modern_kimi_shard_head_info_r9183': ([], B),
    'modern_owner_affinity_report_r2392': ([], B),
    'modern_kimi_engine_destroy_r2475': ([P], I),
    'modern_kimi_engine_error_r2475': ([], B),
}


def declare(lib):
    for name, (args, result) in SIGNATURES.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    return lib


def read_json(lib, name, *args):
    raw = getattr(lib, name)(*args)
    if not raw:
        error = lib.modern_kimi_engine_error_r2475()
        raise RuntimeError((name, error.decode() if error else 'missing JSON'))
    return json.loads(raw)


def check_metadata(config, weight_map):
    """Read checkpoint JSON metadata only; never open a tensor payload."""
    t = config['text_config']
    assert config['model_type'] == 'kimi_k3'
    expected = {'num_hidden_layers': 93, 'hidden_size': 7168,
                'num_attention_heads': 96, 'qk_nope_head_dim': 128,
                'kv_lora_rank': 512, 'qk_rope_head_dim': 64}
    assert all(t[k] == v for k, v in expected.items())
    assert t['max_position_embeddings'] >= 32768
    assert t['linear_attn_config']['num_heads'] == 96
    assert t['linear_attn_config']['head_dim'] == 128
    q = t['quantization_config']
    assert q['format'] == 'mxfp4-pack-quantized'
    assert q['quant_method'] == 'compressed-tensors'
    w = q['config_groups']['group_0']['weights']
    assert w['group_size'] == 32 and w['num_bits'] == 4
    assert w['scale_dtype'] == 'torch.uint8' and w['type'] == 'float'
    assert w['symmetric'] is True and w['dynamic'] is False
    assert weight_map and all(isinstance(k, str) and isinstance(v, str)
                              for k, v in weight_map.items())


def check_observation(value, node, weight_map):
    c = value['capabilities']
    expected = dict(version=1, TP=32, PP=1, C=1, physical_rows=2,
                    state_rows=3, pages=260, page_size=256, context=32768,
                    prompt_chunk=512, decode_width=1, MLA_local_heads=3,
                    KDA_local_heads=3, local_MoE_I=96, local_shared_I=192,
                    head_rows=5120, head_bytes=73400320, stage=0,
                    layer_begin=0, layer_end=93, sampling_node=0,
                    speculation=False, capture=False, replay=False,
                    GPU_qualified=False)
    assert all(c.get(k) == v for k, v in expected.items()), c
    layout = value['layout']
    assert all(layout.get(k) == v for k, v in dict(node=node, local_workers=8,
               TP=32, TP_node_rank=node, PP=1, PP_stage=0, sampling_node=0,
               public_graph_enabled=False, owned_metadata_enabled=False).items())
    cache = value['cache_geometry']
    assert all(cache.get(k) == v for k, v in dict(max_rows=2, pages=260,
               page_size=256, KDA_pool_rows=3, pages_per_request=128,
               max_total_tokens=32768).items()), cache
    assert value['partition'] == [0, 93]
    keys = value['keys']
    assert isinstance(keys, list) and keys and len(keys) == len(set(keys))
    # Exact unchanged public metadata consumer, single PP stage. The old
    # all-stages helper assumes two TP nodes and is deliberately not called.
    value['key_contract'] = summarize_keys(keys, weight_map, 0, [[0, 93]])
    h = value['head']
    assert h['head_kind'] == 'ShardOnlyVocabHeadR9183'
    assert h['expected_CPU_source_registered'] is False  # no weight reads
    assert len(h['owners']) == 8
    devices = set()
    for o in h['owners']:
        d = o['device']; assert d in range(8) and d not in devices
        devices.add(d); rank = node * 8 + d
        assert o['model_constructed'] is True and o['head_constructed'] is True
        assert (o['TP'], o['PP'], o['PP_stage'], o['tp_rank']) == (32, 1, 0, rank)
        assert o['physical_head_shape'] == [5120, 7168]
        assert o['head_parameter_bytes'] == 73400320
        assert (o['global_row_begin'], o['global_row_end_exclusive']) == (rank * 5120, (rank + 1) * 5120)
        assert o['full_GPU_head_retained'] is False
        assert o['head_is_ReplicatedLinear'] is False
        assert o['postload_rows_verified'] is False
        assert o['head_forward_calls'] == o['head_forward_rows'] == 0
    m = value['output_metrics']
    assert all(m[k] == 0 for k in ['engine_calls', 'full_logits_calls',
               'token_only_calls', 'forward_logits_bytes', 'forward_id_bytes'])
    # Affinity is a raw snapshot. Lazy first-submit binding is not forced by
    # constructor-only allocation, nor simulated by binding the calling thread.
    assert len(value['affinity']['owners']) == 8
    return value


def observe(lib, ctx):
    out = {key: read_json(lib, name, ctx) for key, name in {
        'capabilities': 'modern_kimi_long_geometry_capabilities_r453',
        'keys': 'modern_kimi_engine_keys_r2475',
        'layout': 'modern_kimi_engine_layout_r2475',
        'cache_geometry': 'modern_kimi_engine_cache_geometry_r2475',
        'output_metrics': 'modern_kimi_engine_output_metrics_r2475'}.items()}
    out['head'] = read_json(lib, 'modern_kimi_shard_head_info_r9183')
    out['affinity'] = read_json(lib, 'modern_owner_affinity_report_r2392')
    interval = (S * 2)()
    assert lib.modern_kimi_engine_partition_r2475(0, 1, interval) == 0
    out['partition'] = list(interval)
    return out


@contextmanager
def resident_constructor(lib, config, weight_map, *, node, address, port, record):
    """Create/inspect/destroy on one caller thread, exactly once; no reset.

    record(event, payload) must persist before returning. The external owner
    must enforce absolute deadlines, capacity/floor, dependency maps and CCL.
    Exceptions never cause automatic create retry.
    """
    assert type(node) is int and node in range(4)
    assert isinstance(address, str) and address and type(port) is int and 0 < port < 65536
    check_metadata(config, weight_map)
    declare(lib)
    owner = threading.get_ident()
    ctx = None
    record('before_create', {'node': node, 'caller_thread': owner})
    try:
        ctx = lib.modern_kimi_engine_create_long_geometry_r453(
            json.dumps(config, separators=(',', ':')).encode(), node, 4, 1,
            address.encode(), port)
        if not ctx:
            e = lib.modern_kimi_engine_error_r2475()
            raise RuntimeError(e.decode() if e else 'create returned NULL')
        record('created', {'context_address': int(ctx), 'node': node})
        actual = observe(lib, ctx)
        record('constructor_observation', actual)  # save before assertions
        check_observation(actual, node, weight_map)
        yield ctx, actual
    finally:
        if ctx:
            assert threading.get_ident() == owner, 'destroy must use create caller thread'
            rc = lib.modern_kimi_engine_destroy_r2475(ctx)
            record('destroy_returned', {'code': rc, 'node': node})
            assert rc == 0, 'normal destroy failed'
