"""Run stdlib fixtures through the layout guard and native diagnostic call path."""

from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import upstream_layout as probe


class Device:
    index = 1

    def __str__(self):
        return 'cuda:1'


class Tensor:
    device = Device()
    dtype, layout, is_cuda = 'torch.bfloat16', 'strided', True
    _version, offset, requires_grad, grad = 0, 0, False, None

    def __init__(self, shape, tag, *, packed=True):
        self.shape, self.tag, self.packed = shape, tag, packed
        self.pointer = id(self)

    def stride(self):
        return (self.shape[-1]*self.shape[-2], self.shape[-1]+(0 if self.packed else 16), 1)

    def storage_offset(self):
        return self.offset

    def data_ptr(self):
        return self.pointer

    def is_contiguous(self):
        return self.packed

    def element_size(self):
        return 2

    def untyped_storage(self):
        return SimpleNamespace(nbytes=lambda:2**22)

    def detach(self):
        return self

    def requires_grad_(self):
        return self


class Test(unittest.TestCase):
    def setUp(self):
        self.stream = 101
        self.torch = SimpleNamespace(
            cuda=SimpleNamespace(current_device=lambda:1, current_stream=lambda _:SimpleNamespace(cuda_stream=self.stream)),
            bfloat16='torch.bfloat16', strided='strided', isfinite=lambda value:SimpleNamespace(all=lambda:True),
            empty=lambda shape, **_:Tensor(shape, 'scratch'),
            backends=SimpleNamespace(cuda=SimpleNamespace(preferred_blas_library=lambda:'preferred',
                                                         _BlasBackends={'cublas':'preferred'})),
        )

    def test_layout_span_storage_offset_and_overlap(self):
        self.assertEqual(probe.layout_span(probe.DY, (190464,1024,1)), 190464)
        self.assertEqual(probe.layout_span(probe.DY, (190464,1,186)), 190464)
        dy = Tensor(probe.DY, 'gradient', packed=False)
        dy.offset = 16
        info = probe.descriptor(dy)
        span = probe.guard_upstream(info, dy.untyped_storage().nbytes(), 2)
        self.assertGreater(span, probe.PACK_BYTES)
        with self.assertRaisesRegex(ValueError, 'out-of-storage'):
            probe.guard_upstream(info, span, 2)
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            probe.layout_span(probe.DY, (0,0,1))
        with self.assertRaisesRegex(ValueError, 'invalid'):
            probe.layout_span(probe.DY, (0,-1024,1))
        with self.assertRaisesRegex(ValueError, 'fixed BF16'):
            probe.guard_upstream(dict(info, shape=(1,192,1024)), 2**22, 2)

    def test_original_upstream_alias_rejects_owner_stream_and_version_changes(self):
        dy = Tensor(probe.DY, 'gradient', packed=False)
        owner = probe.OwnedUpstream(dy, self.torch)
        self.assertEqual(owner.pointer, dy.data_ptr())
        self.assertEqual(owner.info['stride'], dy.stride())
        self.stream += 1
        with self.assertRaisesRegex(ValueError, 'stream changed'):
            owner.check(self.torch)
        self.stream -= 1
        dy._version += 1
        with self.assertRaisesRegex(ValueError, 'view or owner stream changed'):
            owner.check(self.torch)

    def run_layout(self, *, packed=False, wrong=False):
        x = Tensor(probe.X, 'x')
        dy = Tensor(probe.DY, 'dy', packed=packed)
        y = Tensor(probe.DY, 'y')
        dx = Tensor(probe.X, 'dx')
        calls = []
        def trial(_torch, _linear, _x, _w, input_dy, scratch, arm):
            self.assertIs(input_dy, dy)
            calls.append(arm)
            if arm == 'packed_copy':
                scratch.tag = input_dy.tag
            value = Tensor(probe.X, 'bad-dx') if wrong and len(calls)==7 else dx
            return y, value, dict(forward_event_ms=.09, pack_event_ms=.02 if arm=='packed_copy' else 0.,
                                  dx_event_ms=.20, pack_plus_dx_event_ms=.22 if arm=='packed_copy' else .25,
                                  host_wall_ns=900000 if arm=='packed_copy' else 1000000)
        with (patch.object(probe,'same_bits',side_effect=lambda _,l,r:l.tag==r.tag),
              patch.object(probe,'fingerprint',side_effect=lambda _,value:value.tag)):
            result = probe.evaluate(self.torch, None, x, Tensor(probe.W,'weight'), probe.OwnedUpstream(dy,self.torch),
                                    y, dx, lambda:None, trial=trial)
        return result,calls

    def test_abba_and_copy_timing_with_exact_failure_denominator(self):
        result,calls = self.run_layout()
        self.assertEqual(calls, [arm for arm in probe.ORDER for _ in range(4)])
        self.assertEqual((result['planned_measured_trials'],result['completed_measured_trials']), (12,12))
        self.assertEqual(result['packed_over_native_pairs'], [.88,.88])
        self.assertEqual(result['pack_storage_bytes'],380928)
        self.assertEqual(result['candidate_status'],'HOST_WALL_AND_NEXT_NATIVE_VALIDATION_REQUIRED')
        failed,_ = self.run_layout(wrong=True)
        self.assertEqual((failed['planned_measured_trials'],failed['completed_measured_trials']), (12,4))
        self.assertIs(failed['numerical_gate_passed'],False)
        self.assertEqual(failed['candidate_status'],'INCOMPLETE')
        contiguous,calls = self.run_layout(packed=True)
        self.assertEqual(calls,['native_stride']*4)
        self.assertEqual(contiguous['candidate_status'],'STOP_ALREADY_CONTIGUOUS')

    def test_single_native_call_and_hook_never_replaces_gradient(self):
        class Base:
            microbatch = 1
            operand = weight = upstream = input_grad = None
            def capture_on_view(self, original, _weight, value):
                return original(value)
        x, y = Tensor(probe.X,'x'), Tensor(probe.DY,'y')
        observer = probe.native_observer(Base, self.torch, None)()
        calls = []
        def original(value):
            calls.append(value)
            return y
        self.assertIs(observer.capture_on_view(original,None,x), y)
        self.assertEqual(calls,[x])
        gradient = Tensor(probe.DY,'dy',packed=False)
        self.assertIsNone(observer.keep_upstream(gradient))
        self.assertIs(observer.native_upstream.value,gradient)
        self.assertIsNone(observer.keep_upstream(gradient))
        self.assertEqual(observer.evaluate(10)['candidate_status'],'UNSUPPORTED_NATIVE_UPSTREAM')
        self.assertIsNone(observer.native_upstream)

    def test_actual_timed_path_copy_is_inside_pack_plus_dx_window(self):
        operations = []
        class Event:
            def __init__(self, **_opt):
                self.index = None
            def record(self):
                self.index=len(operations)
                operations.append('mark')
            def synchronize(self):
                operations.append('drain')
            def elapsed_time(self, other):
                return float(other.index-self.index)
        self.torch.cuda.Event = Event
        self.torch.enable_grad = nullcontext
        self.torch.autocast = lambda *_args, **_opts:nullcontext()
        x, w = Tensor(probe.X,'x'),Tensor(probe.W,'w')
        y, upstream = Tensor(probe.DY,'y'),Tensor(probe.DY,'upstream')
        scratch, dx = Tensor(probe.DY,'scratch'),Tensor(probe.X,'dx')
        def linear(given_x, given_w):
            self.assertEqual((given_x,given_w),(x,w))
            operations.append('linear')
            return y
        def copy_from(tensor):
            operations.append('copy')
            self.assertIs(tensor,upstream)
        scratch.copy_ = copy_from
        def backward(o,arg,dy):
            self.assertEqual((o,arg,dy),(y,x,scratch))
            operations.append('backward')
            return (dx,)
        self.torch.autograd = SimpleNamespace(grad=backward)
        _, _, times = probe.timed_trial(self.torch,linear,x,w,upstream,scratch,'packed_copy')
        self.assertEqual(operations,['mark','linear','mark','copy','mark','backward','mark','drain'])
        self.assertEqual(times['pack_event_ms']+times['dx_event_ms'],times['pack_plus_dx_event_ms'])


if __name__ == '__main__':
    unittest.main()
