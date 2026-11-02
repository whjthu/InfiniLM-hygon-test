"""Single actual Runtime, current TP16 rank0 weight slice, NOT TP16 execution.

Historical sixteen recorded rows repeated to512: geometry evidence, not Model prefill.
All numerical/routing state is fixed between arms; the BF16 arm rounds SITU output.
Import, descriptor and execute_calls are CPU safe. main alone loads weights/device.
"""
from pathlib import Path
import ctypes
import hashlib
import json
import os
import time
from plan import records, row_order, descriptor

D = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def execute_calls(pointers, deadline, set_mode, invoke, counter, inspect, observe):
    """Actual production Python->CAPI argument glue, recording leaf on CPU only."""
    result = []
    for record in records():
        args = descriptor(record, pointers[record['fixture']], deadline)
        if set_mode(record['arm']):
            raise RuntimeError('Explicit owner mode refused')
        before = counter()
        tick = time.perf_counter()
        code = invoke(args)
        wall = (time.perf_counter() - tick) * 1000
        if code:
            raise RuntimeError('Complete routedFFN failed; no retry')
        after = counter()
        delta = {k: [b - a for a, b in zip(before[k], after[k])] for k in before}
        metadata, decoded = delta['metadata'], delta['decoder']
        if metadata[0] != 1 or metadata[1] != 512 or metadata[2] <= 0 or metadata[3] != 3 * metadata[2] or metadata[6] != 0:
            raise RuntimeError('Actual complete helper plan did not execute')
        if record['arm']:
            if decoded[0] != metadata[2] or decoded[2] != metadata[2] or decoded[1] != decoded[3] or decoded[1] <= 0:
                raise RuntimeError('Both I192 BF16 W13/W2 decoders must hit')
        elif any(decoded):
            raise RuntimeError('OFF unexpectedly used BF16 decoder')
        row = dict(record, complete_call_wall_ms=wall, counters_delta=delta,
                   dtype='BF16 operands/F32 GEMM outputs + BF16 SITU' if record['arm'] else 'FP32 operands/F32 SITU')
        row.update(inspect(record))
        result.append(row)
        observe(result)
    return result


def main(manifest, output):
    import gc
    import numpy as np
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    m = json.loads(Path(manifest).read_bytes())
    cpu = Path(m['CPU_receipt'])
    assert sha(cpu) == m['CPU_receipt_SHA256']
    c = json.loads(cpu.read_bytes())
    assert c['completed'] and c['passed'] and not c['GPU_initialized'] and not c['Native_dlopen']
    assert sha(c['library']) == c['library_SHA256'] == m['library_SHA256']
    output_path = Path(output)
    assert not output_path.exists()
    q = dict(completed=False, passed=False, actual_GPU=os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID'],
             PID=os.getpid(), GPU_count=1, Runtime_world=1, logical_weight_slice=dict(TP=16, rank=0),
             CCL=False, Model_loaded=False, HTTP=False, geometry_only=True, scope=__doc__,
             source_manifest_SHA256=m['source_manifest_SHA256'], CPU_receipt_SHA256=sha(cpu),
             library_SHA256=sha(c['library']), normal_Runtime_close=False, attempted=0, completed_calls=0)

    def save():
        temp = output_path.with_suffix('.tmp')
        temp.write_text(json.dumps(q, indent=2) + '\n')
        temp.replace(output_path)

    save()
    lib = None
    try:
        lib = ctypes.CDLL(c['library'], mode=ctypes.RTLD_LOCAL)
        lib.kimi401_grouped_mxfp4.argtypes = [ctypes.c_int] * 2 + [ctypes.c_size_t] * 5 + [ctypes.c_void_p] * 9
        lib.kimi401_grouped_mxfp4.restype = ctypes.c_int
        lib.kimi401_set_operand_mode.argtypes = [ctypes.c_int]
        lib.kimi401_set_operand_mode.restype = ctypes.c_int
        lib.kimi401_grouped_error.restype = ctypes.c_char_p
        for name, n in [('kimi401_grouped_metadata_counts_r5858', 7), ('kimi401_decoder_counts', 4)]:
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
            fn.restype = ctypes.c_int
        lib.kimi401_grouped_shutdown.argtypes = []
        lib.kimi401_grouped_shutdown.restype = ctypes.c_int
        # Provider identity resolved inside the actual device container.
        class Dl(ctypes.Structure):
            _fields_ = [('path', ctypes.c_char_p), ('base', ctypes.c_void_p), ('name', ctypes.c_char_p), ('addr', ctypes.c_void_p)]
        dl = ctypes.CDLL(None).dladdr
        dl.argtypes = [ctypes.c_void_p, ctypes.POINTER(Dl)]
        q['provider_symbols'] = {}
        for name in ['kimi401_grouped_mxfp4', 'kimi401_bounded_decode_bf16_r6054', 'modern_grouped_decode_batch_f32_r21']:
            v = Dl()
            assert dl(ctypes.cast(getattr(lib, name), ctypes.c_void_p), ctypes.byref(v))
            actual = Path(v.path.decode()).resolve()
            assert actual == Path(c['library']).resolve()
            q['provider_symbols'][name] = dict(path=str(actual), SHA256=sha(actual))
        model = Path('/shared/Kimi-K3')
        assert sha(model / 'config.json') == '9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213'
        assert sha(model / 'model.safetensors.index.json') == 'a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd'
        mapping = json.loads((model / 'model.safetensors.index.json').read_bytes())['weight_map']
        prefix = 'language_model.model.layers.1.block_sparse_moe.experts.'
        bank_shapes = [(896, 384, 1792), (896, 384, 112), (896, 3584, 96), (896, 3584, 6)]
        hostbanks = [torch.empty(s, dtype=torch.uint8) for s in bank_shapes]
        before = time.monotonic()
        for expert in range(896):
            if time.time() >= m['deadline_unix']:
                raise TimeoutError('Original deadline during checkpoint load')
            loaded = {}
            for part in ['w1', 'w3', 'w2']:
                for name in ['weight_packed', 'weight_scale']:
                    key = prefix + str(expert) + '.' + part + '.' + name
                    with safe_open(str(model / mapping[key]), framework='pt', device='cpu') as f:
                        # Slice the CPU mapped view before cloning; never stage all896 full tensors.
                        view = f.get_slice(key)
                        if part != 'w2':
                            value = view[:192, :]
                        else:
                            value = view[:, :96 if name == 'weight_packed' else 6]
                        loaded[part, name] = value.contiguous()
            vals = [torch.cat([loaded[p, n] for p in ['w1', 'w3']]) for n in ['weight_packed', 'weight_scale']]
            vals += [loaded['w2', n] for n in ['weight_packed', 'weight_scale']]
            for dst, src in zip(hostbanks, vals):
                assert src.dtype == torch.uint8 and dst[expert].shape == src.shape
                dst[expert].copy_(src)
        q['weight_load_seconds'] = time.monotonic() - before
        q['all_loaded_rank0_scales'] = dict(W13_min=int(hostbanks[1].min()), W13_max=int(hostbanks[1].max()),
                                           W2_min=int(hostbanks[3].min()), W2_max=int(hostbanks[3].max()),
                                           count=hostbanks[1].numel() + hostbanks[3].numel())
        assert max(q['all_loaded_rank0_scales']['W13_max'], q['all_loaded_rank0_scales']['W2_max']) <= 252, 'Reject any scale253..255, no partial bank fallback'
        bank_hashes = [hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in hostbanks]
        banks = [v.cuda() for v in hostbanks]
        pins = json.loads((D / 'FIXTURE_PINS.json').read_bytes())
        arrays = []
        for name, dtype, shape in [('rank0-layer1-helper-input-bf16.bin', np.uint16, (16, 3584)),
                                    ('rank0-layer1-router-ids-i32.bin', np.int32, (16, 16)),
                                    ('rank0-layer1-router-scales-f32.bin', np.float32, (16, 16))]:
            p = Path(pins[name]['path'])
            assert sha(p) == pins[name]['SHA256'] and p.stat().st_size == pins[name]['bytes']
            arrays.append(np.frombuffer(p.read_bytes(), dtype=dtype).reshape(shape))
        assert np.isfinite(arrays[2]).all() and (arrays[1] >= 0).all() and (arrays[1] < 896).all()
        inputs, outputs, pointers = [], [], []
        q['fixtures'] = []
        for fixture in range(2):
            expanded = [np.ascontiguousarray(v[row_order(fixture)]) for v in arrays]
            x = torch.from_numpy(expanded[0].view(np.int16)).view(torch.bfloat16).cuda()
            ids = torch.from_numpy(expanded[1]).cuda()
            weights = torch.from_numpy(expanded[2]).cuda()
            # Guarded output tail survives every complete invocation.
            storage = torch.full((512 * 3584 + 256,), 0x3f80, dtype=torch.int16, device='cuda')
            out = storage[:512 * 3584].view(torch.bfloat16).reshape(512, 3584)
            inputs.append([x, ids, weights]); outputs.append([storage, out])
            pointers.append([out.data_ptr(), x.data_ptr(), ids.data_ptr(), weights.data_ptr(), *[v.data_ptr() for v in banks], torch.cuda.current_stream().cuda_stream])
            chosen = np.unique(expanded[1]); q['fixtures'].append(dict(fixture=fixture, shape=[512, 3584],
                input_SHA256=[hashlib.sha256(v.tobytes()).hexdigest() for v in expanded], chosen_experts=chosen.tolist(),
                selected_scale_minmax=[[int(hostbanks[i][chosen].min()), int(hostbanks[i][chosen].max())] for i in [1, 3]],
                source='Historical teacher16 repeated in exactly the two391 row orders; not real512prefill.'))
        torch.cuda.synchronize()
        reference = {}

        def inspect(record):
            f = record['fixture']; current = outputs[f][1].cpu().float().numpy()
            assert np.isfinite(current).all(), 'NaN/Inf stop'
            if f not in reference:
                assert record['arm'] == 0
                reference[f] = current.copy()
            ref = reference[f]
            err = current.astype(np.float64) - ref.astype(np.float64)
            nr = float(np.sqrt(np.mean(err * err)) / max(float(np.sqrt(np.mean(ref.astype(np.float64) ** 2))), 1e-30))
            mx = float(np.max(np.abs(err))); normalized = mx / max(float(np.max(np.abs(ref))), 1e-30)
            assert nr <= 1 / 64 and normalized <= 1 / 8, 'CompleteFFN gross numerical screen'
            readonly = [hashlib.sha256(v.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for v in inputs[f]]
            assert readonly == q['fixtures'][f]['input_SHA256']
            assert bool((outputs[f][0][-256:] == 0x3f80).all())
            return dict(finite=True, relative_RMS=nr, max_abs=mx, maxabs_over_reference_max=normalized,
                        readonly=True, output_tail_canary=True)

        def counter():
            result = {}
            for key, name, n in [('metadata', 'kimi401_grouped_metadata_counts_r5858', 7), ('decoder', 'kimi401_decoder_counts', 4)]:
                values = (ctypes.c_uint64 * n)()
                assert getattr(lib, name)(0, values, n) == 0
                result[key] = list(values)
            return result

        def invoke(args):
            q['attempted'] += 1
            # Poison output outside timer in inspect's predecessor? Set poison in set_mode callback.
            code = lib.kimi401_grouped_mxfp4(*args)
            torch.cuda.synchronize()
            if code:
                raise RuntimeError(lib.kimi401_grouped_error().decode())
            return code

        plan = iter(records())
        def set_mode(arm):
            record = next(plan)
            assert record['arm'] == arm
            outputs[record['fixture']][1].fill_(float('nan'))
            torch.cuda.synchronize()
            return lib.kimi401_set_operand_mode(arm)

        def observe(rows):
            q['trials'] = rows; q['completed_calls'] = len(rows); save()

        q['trials'] = execute_calls(pointers, m['deadline_unix'], set_mode, invoke, counter, inspect, observe)
        assert len(q['trials']) == q['attempted'] == 68
        assert [hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for v in banks] == bank_hashes
        q['all_checkpoint_banks_readonly'] = True
        assert lib.kimi401_grouped_shutdown() == 0
        q['normal_Runtime_close'] = True
        del inputs, outputs, banks, hostbanks
        gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
        check = torch.ones(16, device='cuda'); assert (check + 1).sum().item() == 32
        torch.cuda.synchronize()
        q.update(completed=True, passed=True, Torch_after_Runtime_close=True)
    except BaseException as error:
        q['error'] = repr(error)
        raise
    finally:
        if lib is not None and not q['normal_Runtime_close']:
            try:
                q['failure_cleanup_return'] = lib.kimi401_grouped_shutdown()
                q['normal_Runtime_close'] = q['failure_cleanup_return'] == 0
            except BaseException as cleanup_error:
                q['cleanup_error'] = repr(cleanup_error)
        try:
            save()
        except BaseException as report_error:
            if 'error' not in q:
                raise
            # A report I/O failure must not replace the actual device/primary failure.
            import sys
            print('Final report failed after primary: ' + repr(report_error), file=sys.stderr)


if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2])
