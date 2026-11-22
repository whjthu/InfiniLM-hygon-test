"""Use one genuine native v_proj gradient to separate packing from its dX."""

from hashlib import sha256
import math
from statistics import median
from time import perf_counter_ns, time


X = (1, 186, 4096)
DY = (1, 186, 1024)
W = (1024, 4096)
ORDER = ('native_stride', 'packed_copy', 'packed_copy', 'native_stride')
PACK_BYTES = math.prod(DY)*2
MAX_ACTIVE_SPAN_BYTES = 8*1024**2
MEASURED = 3


def need(ok, message):
    if not ok:
        raise ValueError(message)


def layout_span(shape, stride):
    """A sufficient non-overlapping strided rectangular view check."""
    need(len(shape) == len(stride) and all(type(n) is int and n > 0 for n in shape)
         and all(type(s) is int and s >= 0 for s in stride), 'invalid native stride descriptor')
    span = 1
    for step, extent in sorted((step, extent) for extent, step in zip(shape, stride, strict=True) if extent > 1):
        need(step >= span, 'native upstream layout overlaps or is unproven')
        span += (extent-1)*step
    return span


def descriptor(tensor):
    return dict(shape=tuple(tensor.shape), stride=tuple(tensor.stride()), offset=tensor.storage_offset(),
                pointer_mod256=tensor.data_ptr() % 256, dtype=str(tensor.dtype), device=str(tensor.device),
                contiguous=bool(tensor.is_contiguous()))


def guard_upstream(info, base_storage_bytes, element_bytes):
    need(info['shape'] == DY and info['dtype'] == 'torch.bfloat16' and element_bytes == 2
         and info['device'].startswith('cuda:') and type(info['offset']) is int and info['offset'] >= 0,
         'native v_proj upstream is outside the fixed BF16 GPU shape')
    span_bytes = layout_span(info['shape'], info['stride'])*element_bytes
    need(span_bytes <= MAX_ACTIVE_SPAN_BYTES and info['offset']*element_bytes+span_bytes <= base_storage_bytes,
         'unbounded or out-of-storage native upstream view')
    return span_bytes


class OwnedUpstream:
    """Retain the original native grad's storage and verify its lifetime in process."""

    def __init__(self, tensor, torch):
        need(tensor.layout == torch.strided, 'unsupported native upstream storage type')
        self.info = descriptor(tensor)
        self.active_span_bytes = guard_upstream(self.info, tensor.untyped_storage().nbytes(), tensor.element_size())
        self.value = tensor.detach()  # No clone, which could silently change the native stride.
        self.version = tensor._version
        self.pointer = tensor.data_ptr()
        self.stream = torch.cuda.current_stream(tensor.device).cuda_stream
        self.check(torch)

    def check(self, torch):
        need(self.value._version == self.version and self.value.data_ptr() == self.pointer
             and descriptor(self.value) == self.info
             and torch.cuda.current_stream(self.value.device).cuda_stream == self.stream,
             'native upstream view or owner stream changed')


def same_bits(torch, lhs, rhs):
    return bool(torch.equal(lhs.detach().contiguous().view(torch.int16), rhs.detach().contiguous().view(torch.int16)))


def fingerprint(torch, value):
    flat = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return sha256(memoryview(flat)).hexdigest()


def timed_trial(torch, linear, operand, weight, upstream, workspace, arm):
    start, after_forward, after_pack, done = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    x = operand.detach().requires_grad_()
    wall = perf_counter_ns()
    with torch.enable_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        start.record()
        output = linear(x, weight)
        after_forward.record()
        dy = upstream
        if arm == 'packed_copy':
            workspace.copy_(upstream)
            dy = workspace
        after_pack.record()
        dx, = torch.autograd.grad(output, x, dy)
        done.record()
        done.synchronize()
    elapsed = perf_counter_ns()-wall
    timing = dict(forward_event_ms=start.elapsed_time(after_forward),
                  pack_event_ms=after_forward.elapsed_time(after_pack) if arm == 'packed_copy' else 0.,
                  dx_event_ms=after_pack.elapsed_time(done),
                  pack_plus_dx_event_ms=after_forward.elapsed_time(done), host_wall_ns=elapsed)
    need(x.grad is None and output.dtype == dx.dtype == torch.bfloat16
         and tuple(output.shape) == DY and tuple(dx.shape) == X
         and all(math.isfinite(value) and value >= 0 for value in timing.values())
         and min(timing['forward_event_ms'], timing['dx_event_ms'], timing['pack_plus_dx_event_ms'], elapsed) > 0,
         'wrong autograd result or timing boundary')
    return output, dx, timing


def evaluate(torch, linear, operand, weight, upstream, native_output, native_dx, check_time, *, trial=timed_trial):
    """A read-only diagnostic after one naturally scheduled native backward."""
    need(operand.is_cuda and tuple(operand.shape) == X and operand.is_contiguous()
         and tuple(weight.shape) == W and weight.is_contiguous() and not weight.requires_grad
         and tuple(native_dx.shape) == X and tuple(native_output.shape) == DY
         and weight.device == operand.device == upstream.value.device == native_dx.device == native_output.device
         and torch.cuda.current_device() == operand.device.index
         and operand.dtype == weight.dtype == native_dx.dtype == upstream.value.dtype == native_output.dtype == torch.bfloat16,
         'wrong native frozen projection, upstream or dX')
    need(torch.backends.cuda.preferred_blas_library() == torch.backends.cuda._BlasBackends['cublas'],
         'original inherited conventional BLAS preference unavailable')
    check_time()
    upstream.check(torch)
    if upstream.info['contiguous']:
        order = ('native_stride',)
        workspace = None
    else:
        order = ORDER
        workspace = torch.empty(DY, dtype=torch.bfloat16, device=operand.device)
        need(workspace.is_contiguous() and workspace.data_ptr() != upstream.pointer, 'invalid dedicated pack storage')
    owners = (operand, weight, native_output, native_dx)
    owner_versions = tuple(tensor._version for tensor in owners)
    pin_tensors = (operand, upstream.value, native_output, native_dx)
    pins = tuple(fingerprint(torch, tensor) for tensor in pin_tensors)
    planned = len(order)*MEASURED
    result = dict(site='decoder_06.v_proj', visit_kind='one_new_native_backward', upstream=upstream.info,
                  active_span_bytes=upstream.active_span_bytes, input_SHA256=pins[0], upstream_SHA256=pins[1],
                  baseline_output_SHA256=pins[2], baseline_dx_SHA256=pins[3],
                  pack_storage_bytes=PACK_BYTES if workspace is not None else 0,
                  planned_measured_trials=planned, completed_measured_trials=0,
                  numerical_gate_passed=False, candidate_status='INCOMPLETE', scope='isolated native training leaf',
                  no_grad_accumulation_in_diagnostic=True, formal_training_qualified=False, arms=[])
    try:
        for arm in order:
            row = dict(arm=arm, planned_trials=MEASURED, timings=[])
            result['arms'].append(row)
            for number in range(MEASURED+1):
                check_time()
                upstream.check(torch)
                output, dx, timing = trial(torch, linear, operand, weight, upstream.value, workspace, arm)
                need(bool(torch.isfinite(output).all()) and bool(torch.isfinite(dx).all())
                     and same_bits(torch, output, native_output) and same_bits(torch, dx, native_dx),
                     'diagnostic forward or dX differs from actual native edge')
                if arm == 'packed_copy':
                    need(same_bits(torch, workspace, upstream.value), 'candidate pack changed the actual upstream gradient')
                if number != 0:
                    row['timings'].append(timing)
                    result['completed_measured_trials'] += 1
            row['median'] = {key: median(sample[key] for sample in row['timings']) for key in row['timings'][0]}
        check_time()
        upstream.check(torch)
        need(owner_versions == tuple(tensor._version for tensor in owners)
             and pins == tuple(fingerprint(torch, tensor) for tensor in pin_tensors),
             'native input, upstream or frozen weight version changed')
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        result['failure'] = dict(type=type(error).__name__, detail=str(error)[:180])
        return result
    result['numerical_gate_passed'] = True
    if workspace is None:
        result['candidate_status'] = 'STOP_ALREADY_CONTIGUOUS'
    else:
        baseline = result['arms']
        ratios = [baseline[b]['median']['pack_plus_dx_event_ms']/baseline[a]['median']['pack_plus_dx_event_ms']
                  for a,b in ((0,1),(3,2))]
        result['packed_over_native_pairs'] = ratios
        result['packed_over_native_host_wall_pairs'] = [
            baseline[b]['median']['host_wall_ns']/baseline[a]['median']['host_wall_ns'] for a,b in ((0,1),(3,2))]
        result['candidate_status'] = ('HOST_WALL_AND_NEXT_NATIVE_VALIDATION_REQUIRED'
                                      if max(ratios) <= .95 else 'STOP_WITH_NATIVE_STRIDE')
    return result


def native_observer(base, torch, linear):
    """Subclass the exact existing ProjectionInputGradient in a new caller."""
    class UpstreamLayoutObserver(base):
        native_output = None
        native_output_version = None
        native_upstream = None
        rejection = None

        def capture_on_view(self, original, weight, value):
            result = super().capture_on_view(original, weight, value)
            self.native_output = result.detach()
            self.native_output_version = result._version
            return result

        def keep_upstream(self, grad):
            try:
                need(self.microbatch == 1 and self.native_upstream is None, 'duplicate or foreign upstream hook')
                self.native_upstream = OwnedUpstream(grad, torch)
            except (RuntimeError, TypeError, ValueError) as error:
                self.rejection = dict(type=type(error).__name__, detail=str(error)[:180])
            return None  # Autograd must retain the original gradient.

        def evaluate(self, deadline):
            def check():
                if time() >= deadline:
                    raise TimeoutError('native upstream-layout diagnostic deadline')

            try:
                if self.rejection is not None:
                    return dict(candidate_status='UNSUPPORTED_NATIVE_UPSTREAM', planned_measured_trials=0,
                                completed_measured_trials=0, failure=self.rejection)
                need(self.native_upstream is not None and self.native_output is not None
                     and self.native_output._version == self.native_output_version,
                     'original native output or upstream is absent or changed')
                return evaluate(torch, linear, self.operand, self.weight, self.native_upstream, self.native_output,
                                self.input_grad, check)
            finally:
                self.native_output = self.native_upstream = None
                self.operand = self.weight = self.upstream = self.input_grad = None

    return UpstreamLayoutObserver
