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
from plan import records, row_order, descriptor, pointer_payload

D = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
CAPI_ARGTYPES = [ctypes.c_int] * 2 + [ctypes.c_size_t] * 5 + [ctypes.c_void_p] * 9


def bind_complete_capi(fn):
    fn.argtypes = list(CAPI_ARGTYPES)
    fn.restype = ctypes.c_int
    return fn


def load_prepared_assets(resolve=None):
    """Shared actual-main preflight, before torch/device/weight load; cache small arrays."""
    import numpy as np
    resolve = resolve or Path
    pins = json.loads((D / 'FIXTURE_PINS.json').read_bytes())
    arrays = []
    for name, dtype, shape in [('rank0-layer1-helper-input-bf16.bin', np.uint16, (16, 3584)),
                               ('rank0-layer1-router-ids-i32.bin', np.int32, (16, 16)),
                               ('rank0-layer1-router-scales-f32.bin', np.float32, (16, 16))]:
        p = Path(resolve(pins[name]['path']))
        raw = p.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == pins[name]['SHA256'] and len(raw) == pins[name]['bytes']
        arrays.append(np.frombuffer(raw, dtype=dtype).reshape(shape))
    assert np.isfinite(arrays[2]).all() and (arrays[1] >= 0).all() and (arrays[1] < 896).all()
    return arrays


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
        after = counter()
        delta = {k: [b - a for a, b in zip(before[k], after[k])] for k in before}
        metadata, decoded = delta['metadata'], delta['decoder']
        counter_pass = metadata[0] == 1 and metadata[1] == 512 and metadata[2] > 0 and metadata[3] == 3 * metadata[2] and metadata[6] == 0
        if record['arm']:
            counter_pass = counter_pass and decoded[0] == metadata[2] and decoded[2] == metadata[2] and decoded[1] == decoded[3] and decoded[1] > 0
        else:
            counter_pass = counter_pass and not any(decoded)
        row = dict(record, complete_call_wall_ms=wall, counters_delta=delta, counter_pass=bool(counter_pass),
                   CAPI_returncode=code, CAPI_completed=code == 0,
                   dtype='BF16 operands/F32 GEMM outputs + BF16 SITU' if record['arm'] else 'FP32 operands/F32 SITU')
        row.update(finite=False, relative_RMS=None, max_abs=None, maxabs_over_reference_max=None,
                   readonly=None, output_tail_canary=None, numeric_pass=False)
        if code == 0:
            try:
                row.update(inspect(record))
            except BaseException as error:
                row['inspection_error'] = repr(error)
        row['passed'] = bool(code == 0 and counter_pass and row['numeric_pass'] and row['finite'] and row['readonly'] and row['output_tail_canary'])
        result.append(row)
        primary = None if row['passed'] else RuntimeError('Completed call failed qualification; saved metrics, no retry: ' + str(record))
        try:
            observe(result)
        except BaseException as report_error:
            if primary is not None:
                raise primary from report_error
            raise
        if primary is not None:
            raise primary
    return result


def main(manifest, output, prepared_assets=None, asset_resolver=None):
    # This same main boundary is exercised with actual staged files on CPU.
    # The entry may pass the exact prevalidated arrays, avoiding any late reread.
    arrays = load_prepared_assets(asset_resolver) if prepared_assets is None else prepared_assets
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
             library_SHA256=sha(c['library']), normal_Runtime_close=False, attempted=0, CAPI_returned=0,
             completed_calls=0, qualification_passed_calls=0)

    def save():
        temp = output_path.with_suffix('.tmp')
        temp.write_text(json.dumps(q, indent=2) + '\n')
        temp.replace(output_path)

    save()
    lib = None
    try:
        lib = ctypes.CDLL(c['library'], mode=ctypes.RTLD_LOCAL)
        bind_complete_capi(lib.kimi401_grouped_mxfp4)
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
        for name in ['kimi401_grouped_mxfp4', 'kimi401_bounded_decode_bf16_r6054', 'modern_grouped_decode_batch_f32_r21', 'cublasGemmStridedBatchedEx']:
            v = Dl()
            assert dl(ctypes.cast(getattr(lib, name), ctypes.c_void_p), ctypes.byref(v))
            actual = Path(v.path.decode()).resolve()
            if name.startswith('cublas'):
                assert sha(actual) == 'c06fcaa76e04342e10034e149fc14a3cef79938b6a5867c38aced26a40108c2a'
            else:
                assert actual == Path(c['library']).resolve()
            q['provider_symbols'][name] = dict(path=str(actual), SHA256=sha(actual))
        q['current_dependencies_mapped'] = {}
        for expected in ['/refactor/install/hygon-kimi-5f91ad7-r3672/lib/libinfiniops.so',
                         '/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/infinilm/_lib/libinfinilm_runtime.so']:
            mapped = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if line.split()[-1].endswith(Path(expected).name)}
            assert any(Path(p).resolve() == Path(expected).resolve() for p in mapped)
            assert sha(expected) == c['actual_dynamic_dependencies_SHA256'][expected]
            q['current_dependencies_mapped'][expected] = sha(expected)
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
            pointers.append(pointer_payload(out, x, ids, weights, banks, torch.cuda.current_stream()))
            chosen = np.unique(expanded[1]); q['fixtures'].append(dict(fixture=fixture, shape=[512, 3584],
                input_SHA256=[hashlib.sha256(v.tobytes()).hexdigest() for v in expanded], chosen_experts=chosen.tolist(),
                selected_scale_minmax=[[int(hostbanks[i][chosen].min()), int(hostbanks[i][chosen].max())] for i in [1, 3]],
                source='Historical teacher16 repeated in exactly the two391 row orders; not real512prefill.'))
        torch.cuda.synchronize()
        reference = {}

        def inspect(record):
            f = record['fixture']; current = outputs[f][1].cpu().float().numpy()
            finite = bool(np.isfinite(current).all())
            if f not in reference and finite and record['arm'] == 0:
                reference[f] = current.copy()
            nr = mx = normalized = None
            if finite and f in reference:
                ref = reference[f]
                err = current.astype(np.float64) - ref.astype(np.float64)
                nr = float(np.sqrt(np.mean(err * err)) / max(float(np.sqrt(np.mean(ref.astype(np.float64) ** 2))), 1e-30))
                mx = float(np.max(np.abs(err))); normalized = mx / max(float(np.max(np.abs(ref))), 1e-30)
            readonly = [hashlib.sha256(v.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for v in inputs[f]]
            return dict(finite=finite, relative_RMS=nr, max_abs=mx, maxabs_over_reference_max=normalized,
                        numeric_pass=finite and nr is not None and nr <= 1 / 64 and normalized <= 1 / 8,
                        readonly=readonly == q['fixtures'][f]['input_SHA256'],
                        output_tail_canary=bool((outputs[f][0][-256:] == 0x3f80).all()))

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
            q['CAPI_returned'] += 1
            q['completed_calls'] += code == 0
            if code: q['last_CAPI_error'] = lib.kimi401_grouped_error().decode()
            return code

        plan = iter(records())
        def set_mode(arm):
            record = next(plan)
            assert record['arm'] == arm
            outputs[record['fixture']][1].fill_(float('nan'))
            torch.cuda.synchronize()
            return lib.kimi401_set_operand_mode(arm)

        def observe(rows):
            q['trials'] = rows; q['qualification_passed_calls'] = sum(r['passed'] for r in rows); save()

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
