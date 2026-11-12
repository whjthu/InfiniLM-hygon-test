"""Observed-shape HIP SDK diagnostic; synthetic inputs never qualify a model."""
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
import math
from pathlib import Path
import time
import traceback


def validate_binding(binding):
    if binding.get('platform') != 'hip' or binding.get('logical_device_ids') != [0]:
        raise ValueError('one explicitly bound HIP device required')
    if not isinstance(binding.get('GPU_name'), str) or not binding['GPU_name']:
        raise ValueError('actual admitted GPU identity required')
    if set(binding.get('provider_files', {})) != {'initializer', 'interface', 'triton_interface', 'extension'}:
        raise ValueError('complete actual SDK source/extension identities required')
    for pin in binding['provider_files'].values():
        if not isinstance(pin.get('bytes'), int) or pin['bytes'] <= 0 or len(pin.get('SHA256', '')) != 64:
            raise ValueError('complete provider byte identity required')
        int(pin['SHA256'], 16)
    cases = binding.get('cases')
    if not isinstance(cases, list) or len(cases) != 3:
        raise ValueError('all three observed shapes required')
    for seq, row in zip((32, 256, 288), cases):
        if row['shape'] != [1, seq, 30, 128] or row['stride'] != [seq * 3840, 3840, 128, 1]:
            raise ValueError('observed shape or contiguous BNHD stride differs')
        if len(row['input_SHA256']) != 4 or any(len(h) != 64 for h in row['input_SHA256']):
            raise ValueError('original A100 controlled input identities required')
    if binding.get('sampled_free_floor_bytes') != 16 * 1024**3:
        raise ValueError('SDK free-memory floor differs')
    if binding.get('max_abs_tolerance') != 0.001 or binding.get('relative_l2_tolerance') != 0.02:
        raise ValueError('preregistered component diagnostic thresholds differ')


def cpu_reference(inputs, torch):
    """Independent full dense noncausal attention in CPU F64, including gradients."""
    if len(inputs) != 4 or any(t.device.type != 'cpu' for t in inputs):
        raise ValueError('mathematical reference must remain on CPU')
    live = [t.permute(0, 2, 1, 3).to(dtype=torch.float64).requires_grad_() for t in inputs[:3]]
    q, k, v = live
    probability = torch.softmax((q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5, dim=-1)
    value = probability @ v
    value.backward(inputs[3].permute(0, 2, 1, 3).to(dtype=torch.float64))
    return [value.detach().permute(0, 2, 1, 3).contiguous()] + [
        t.grad.detach().permute(0, 2, 1, 3).contiguous() for t in live]


def compare_reference(actual, reference, torch):
    if len(actual) != 4 or len(reference) != 4:
        raise ValueError('complete output and three-gradient reference required')
    rows = []
    for x, y in zip(actual, reference):
        if x.device.type != 'cpu' or y.device.type != 'cpu' or x.shape != y.shape:
            raise ValueError('output/reference representation differs')
        delta = x.to(dtype=torch.float64) - y
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
            raise ValueError('nonfinite SDK/reference result')
        norm = float(torch.linalg.vector_norm(y))
        if norm == 0:
            raise ValueError('degenerate mathematical reference')
        rows.append(dict(max_abs=float(delta.abs().max()), relative_l2=float(torch.linalg.vector_norm(delta)) / norm,
                         elements=x.numel(), actual_dtype=str(x.dtype), reference_dtype=str(y.dtype)))
    return rows


def run(output, binding, *, epoch, deadline):
    validate_binding(binding)
    if (not all(math.isfinite(v) for v in (epoch, deadline)) or not 0 < deadline - epoch <= 600
            or not epoch <= time.time() < deadline):
        raise ValueError('finite original SDK epoch required')
    anchor = time.monotonic()
    remaining = deadline - time.time()
    def check():
        if min(deadline - time.time(), remaining - (time.monotonic() - anchor)) <= 0:
            raise TimeoutError('original SDK epoch exhausted')
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    plan = [dict(case=seq, arm=arm, repeat=repeat) for seq in (32, 256, 288)
            for arm, repeat in [('default', 0), ('deterministic', 0), ('deterministic', 1), ('deterministic', 2)]]
    events = []
    artifacts = {}
    rows = []
    sampled_memory = []
    counters = dict(forward_attempted=0, forward_returned=0, backward_attempted=0, backward_synchronized=0)
    def progress(event):
        events.append(event)
        (output / 'progress.json').write_text(json.dumps(dict(plan=plan, events=events, counters=counters), indent=2))
    def save(name, tensors, torch):
        path = output / name
        if path.exists():
            raise FileExistsError('private SDK artifact already exists')
        torch.save(tensors, path)
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for raw in iter(lambda: stream.read(4 * 1024**2), b''):
                h.update(raw)
        artifacts[name] = dict(bytes=path.stat().st_size, SHA256=h.hexdigest())
        if len(artifacts) > 18 or sum(pin['bytes'] for pin in artifacts.values()) > 512 * 1024**2:
            raise ValueError('private diagnostic artifact budget exceeded')
    torch = None
    phase = 'provider-files'
    try:
        for pin in binding['provider_files'].values():
            check()
            path = Path(pin['path'])
            if path.is_symlink() or path.stat().st_size != pin['bytes']:
                raise ValueError('actual SDK provider extent differs')
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for raw in iter(lambda: stream.read(4 * 1024**2), b''):
                    check()
                    digest.update(raw)
            if digest.hexdigest() != pin['SHA256']:
                raise ValueError('actual SDK provider SHA differs')
        import torch
        if torch.version.hip != binding['HIP'] or metadata.version('flash-attn') != binding['flash_version']:
            raise ValueError('actual HIP/FlashAttention versions differ')
        generator = torch.Generator(device='cpu')
        generator.manual_seed(20261113)
        inputs_by_shape = []
        for case in binding['cases']:
            check()
            inputs = [(torch.randn(tuple(case['shape']), generator=generator) * .1).to(torch.bfloat16) for _ in range(4)]
            hashes = [hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest() for t in inputs]
            if hashes != case['input_SHA256'] or any(list(t.stride()) != case['stride'] for t in inputs):
                raise ValueError('CPU-generated controlled input differs from original A100 bytes')
            inputs_by_shape.append(inputs)
        phase = 'actual-owned-device-and-exports'
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != binding['GPU_name']:
            raise ValueError('actual admitted accelerator differs')
        torch.cuda.set_device(0)
        root = importlib.import_module('flash_attn')
        interface = importlib.import_module('flash_attn.flash_attn_interface')
        extension = importlib.import_module('flash_attn_2_cuda')
        function = root.flash_attn_func
        if (function is not interface.flash_attn_func
                or Path(inspect.getsourcefile(function)).resolve() != Path(binding['provider_files']['interface']['path']).resolve()
                or Path(root.__file__).resolve() != Path(binding['provider_files']['initializer']['path']).resolve()
                or Path(extension.__file__).resolve() != Path(binding['provider_files']['extension']['path']).resolve()):
            raise ValueError('actual root API or loaded extension provider differs')
        signature = inspect.signature(function)
        if signature.parameters['deterministic'].default is not False:
            raise ValueError('actual deterministic backward API differs')
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        def memory(label):
            free, total = torch.cuda.mem_get_info()
            sampled_memory.append(dict(label=label, free_bytes=free, total_bytes=total))
            if free < binding['sampled_free_floor_bytes']:
                raise ValueError('sampled device free-memory floor violated')
        for case, inputs in zip(binding['cases'], inputs_by_shape):
            seq = case['shape'][1]
            phase = 'CPU-F64-reference-%d' % seq
            check()
            reference = cpu_reference(inputs, torch)
            save('reference-%d.private.pt' % seq, reference, torch)
            save('inputs-%d.private.pt' % seq, inputs, torch)
            outcomes = []
            case_row = dict(shape=case['shape'], stride=case['stride'], input_SHA256=case['input_SHA256'], outcomes=outcomes, complete=False)
            rows.append(case_row)
            baseline = None
            first_deterministic = None
            for arm, repeat in [('default', 0), ('deterministic', 0), ('deterministic', 1), ('deterministic', 2)]:
                phase = 'SDK-%d-%s-%d' % (seq, arm, repeat)
                check()
                memory(phase + '-before')
                live = [t.to('cuda').requires_grad_() for t in inputs[:3]]
                grad = inputs[3].to('cuda')
                memory(phase + '-inputs')
                torch.cuda.synchronize()
                counters['forward_attempted'] += 1
                progress(dict(phase=phase, event='forward.enter'))
                started = time.monotonic()
                value = function(*live, dropout_p=0., causal=False, deterministic=arm == 'deterministic')
                counters['forward_returned'] += 1
                counters['backward_attempted'] += 1
                value.backward(grad)
                torch.cuda.synchronize()
                wall = time.monotonic() - started
                counters['backward_synchronized'] += 1
                actual = [value.detach().cpu()] + [t.grad.detach().cpu() for t in live]
                metrics = compare_reference(actual, reference, torch)
                passed = all(x['max_abs'] <= binding['max_abs_tolerance'] and x['relative_l2'] <= binding['relative_l2_tolerance'] for x in metrics)
                equal = None
                if arm == 'default':
                    baseline = actual
                elif first_deterministic is None:
                    first_deterministic = actual
                else:
                    equal = all(torch.equal(x, y) for x, y in zip(actual, first_deterministic))
                    passed = passed and equal
                save('sdk-%d-%s-%d.private.pt' % (seq, arm, repeat), actual, torch)
                outcomes.append(dict(arm=arm, repeat=repeat, CPU_F64_metrics=metrics, repeat_exact=equal, passed=passed, inclusive_call_seconds=wall))
                progress(dict(phase=phase, event='backward.synchronized', passed=passed))
                del value, live, grad
                torch.cuda.empty_cache()
                memory(phase + '-returned')
                if not passed:
                    raise ValueError('SDK component numerical/repeat gate failed')
            case_row.update(complete=True, forward_default_vs_deterministic_exact=torch.equal(baseline[0], first_deterministic[0]))
        result = dict(passed=True, state='OBSERVED_SHAPE_HIP_SDK_NUMERICAL_AND_REPEAT_PASS', cases=rows,
                      actual_root_signature=str(signature), peak_Torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_Torch_reserved_bytes=torch.cuda.max_memory_reserved(), numerical_scope='controlled synthetic inputs; CPU F64 reference is on CPU only')
    except BaseException as error:
        (output / 'failure.private.txt').write_text(traceback.format_exc())
        result = dict(passed=False, phase=phase, error_type=type(error).__name__, cases=rows)
    result.update(plan=plan, events=events, counters=counters, artifacts=artifacts, sampled_memory=sampled_memory,
                  memory_scope='sampled free-memory boundary and Torch allocator counters; not an observed hardware hard peak',
                  epoch=epoch, deadline=deadline, wall_seconds=time.monotonic() - anchor,
                  actual_native_model_loads=0, actual_native_optimizer_updates=0, model_numerical_qualified=False,
                  binding_canonical_SHA256=hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                  model_attention_policy_qualified=False, formal_performance=False, Model_GO=False)
    with (output / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result
