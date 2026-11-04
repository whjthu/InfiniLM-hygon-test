"""New caller test only: actual ctypes/libffi, recording Native leaves."""
import ctypes as C
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import constructor as target


def metadata():
    return {'model_type': 'kimi_k3', 'text_config': {
        'num_hidden_layers': 93, 'hidden_size': 7168, 'num_attention_heads': 96,
        'qk_nope_head_dim': 128, 'kv_lora_rank': 512, 'qk_rope_head_dim': 64,
        'max_position_embeddings': 32768,
        'linear_attn_config': {'num_heads': 96, 'head_dim': 128},
        'quantization_config': {'format': 'mxfp4-pack-quantized',
            'quant_method': 'compressed-tensors', 'config_groups': {'group_0': {
                'weights': {'group_size': 32, 'num_bits': 4,
                    'scale_dtype': 'torch.uint8', 'type': 'float',
                    'symmetric': True, 'dynamic': False}}}}}}


def observation(node):
    caps = dict(version=1, TP=32, PP=1, C=1, physical_rows=2, state_rows=3,
                pages=260, page_size=256, context=32768, prompt_chunk=512,
                decode_width=1, MLA_local_heads=3, KDA_local_heads=3,
                local_MoE_I=96, local_shared_I=192, head_rows=5120,
                head_bytes=73400320, stage=0, layer_begin=0, layer_end=93,
                sampling_node=0, speculation=False, capture=False, replay=False,
                GPU_qualified=False)
    owners = []
    for d in range(8):
        rank = node * 8 + d
        owners.append(dict(device=d, TP=32, PP=1, PP_stage=0, tp_rank=rank,
            model_constructed=True, head_constructed=True,
            physical_head_shape=[5120, 7168], head_parameter_bytes=73400320,
            global_row_begin=rank * 5120, global_row_end_exclusive=(rank + 1) * 5120,
            full_GPU_head_retained=False, head_is_ReplicatedLinear=False,
            postload_rows_verified=False, head_forward_calls=0, head_forward_rows=0))
    return dict(capabilities=caps, layout=dict(node=node, local_workers=8,
            TP=32, TP_node_rank=node, PP=1, PP_stage=0, sampling_node=0,
            public_graph_enabled=False, owned_metadata_enabled=False),
        cache_geometry=dict(max_rows=2, pages=260, page_size=256, KDA_pool_rows=3,
            pages_per_request=128, max_total_tokens=32768),
        keys=['language_model.model.layers.%d.weight' % i for i in range(93)] +
             ['language_model.model.embed_tokens.weight', 'language_model.lm_head.weight'],
        head=dict(head_kind='ShardOnlyVocabHeadR9183',
            expected_CPU_source_registered=False, owners=owners),
        affinity=dict(owners=[dict(device=d, changes=0, verified_mask='') for d in range(8)]),
        output_metrics=dict(engine_calls=0, full_logits_calls=0, token_only_calls=0,
            forward_logits_bytes=0, forward_id_bytes=0), partition=[0, 93])


def ffi_leaves(node=3, null=False, invalid=False):
    lib = SimpleNamespace()
    keep = []
    calls = []
    ctx = 0x12345678ABCDEF01
    actual = observation(node)
    if invalid:
        actual['capabilities']['context'] = 1024
    mapping = { 'modern_kimi_long_geometry_capabilities_r453': 'capabilities',
        'modern_kimi_engine_keys_r2475': 'keys',
        'modern_kimi_engine_layout_r2475': 'layout',
        'modern_kimi_engine_cache_geometry_r2475': 'cache_geometry',
        'modern_kimi_engine_output_metrics_r2475': 'output_metrics',
        'modern_kimi_shard_head_info_r9183': 'head',
        'modern_owner_affinity_report_r2392': 'affinity'}

    def leaf(name, *args):
        calls.append((name, list(args)))
        if name == 'modern_kimi_engine_create_long_geometry_r453':
            assert args[1:4] == (node, 4, 1) and args[5] == 26855
            assert json.loads(args[0]) == metadata()
            return 0 if null else ctx
        if name == 'modern_kimi_engine_partition_r2475':
            assert args[:2] == (0, 1)
            args[2][0], args[2][1] = 0, 93
            return 0
        if name == 'modern_kimi_engine_destroy_r2475':
            assert args[0] == ctx
            return 0
        if name == 'modern_kimi_engine_error_r2475':
            payload = b'recorded create failure'
        else:
            if args:
                assert args[0] == ctx, 'high context pointer truncated'
            payload = json.dumps(actual[mapping[name]]).encode()
        buf = C.create_string_buffer(payload)
        keep.append(buf)
        return C.addressof(buf)

    for name, (args, result) in target.SIGNATURES.items():
        # Actual native-address calls through libffi; c_void_p callback return
        # owns retained buffers for C-string results, avoiding PythonFakeLib.
        callback_result = C.c_void_p if result is C.c_char_p else result
        callback = C.CFUNCTYPE(callback_result, *args)(lambda *a, n=name: leaf(n, *a))
        fn = C.CFUNCTYPE(callback_result, *args)(C.cast(callback, C.c_void_p).value)
        keep.extend([callback, fn])
        setattr(lib, name, fn)
    lib._keep = keep
    return lib, calls, ctx


def run():
    cases = []
    for label, null, invalid, body_failure in [('success', False, False, False),
            ('bad_geometry_saved_before_stop', False, True, False),
            ('body_failure_destroyed', False, False, True),
            ('null_create_no_destroy', True, False, False)]:
        lib, calls, pointer = ffi_leaves(null=null, invalid=invalid)
        events = []
        keys = observation(3)['keys']
        failed = False
        try:
            with target.resident_constructor(lib, metadata(), dict.fromkeys(keys, 'metadata-only'),
                    node=3, address='recording-head', port=26855,
                    record=lambda event, payload: events.append((event, payload))) as (ctx, _):
                assert ctx == pointer
                if body_failure:
                    raise ValueError('recorded consumer failure')
        except (AssertionError, RuntimeError, ValueError):
            failed = True
        assert failed == (null or invalid or body_failure)
        names = [name for name, _ in calls]
        assert names.count('modern_kimi_engine_create_long_geometry_r453') == 1
        assert names.count('modern_kimi_engine_destroy_r2475') == (0 if null else 1)
        if invalid:
            assert any(event == 'constructor_observation' and payload['capabilities']['context'] == 1024
                       for event, payload in events)
        assert all(not any(word in name for word in ['forward', 'load_', 'finish_', 'reset_', 'batch_'])
                   for name in names)
        cases.append(dict(case=label, passed=True, calls=names,
                          context_high_bits_preserved=not null, actual_Native=False))
    return dict(passed=True, scope='new local constructor consumer only', cases=cases,
        actual_libffi=True, Native_dlopen=False, Runtime_created=False,
        weights=False, GPU=False, Model_READY=False,
        source_SHA256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__), Path(target.__file__)]})


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
