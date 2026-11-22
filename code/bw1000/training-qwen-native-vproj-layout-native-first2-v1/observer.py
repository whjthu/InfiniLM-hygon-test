"""Observe one naturally occurring Qwen projection edge during native training."""

import math
import time


MAX_MICROBATCHES = 4
MAX_DIAGNOSTIC_SECONDS = 20
SITE = 'decoder_06.v_proj'


def error_info(error):
    return dict(type=type(error).__name__, detail=str(error)[:160])


class Slot:
    def __init__(self, tensor):
        self.tensor = tensor.detach()
        self.version = tensor._version
        self.pointer = tensor.data_ptr()

    def get(self):
        if self.tensor._version != self.version or self.tensor.data_ptr() != self.pointer:
            raise ValueError('observed native tensor no longer owns the same version and pointer')
        return self.tensor


class Edge:
    def __init__(self, torch, layout, value, weight):
        self.torch, self.layout = torch, layout
        self.x, self.weight = Slot(value), Slot(weight)
        self.y = self.dy = self.dx = self.error = None
        self.hooks = []
        self.closed = False
        self.dy_calls = self.dx_calls = 0

    def save_dx(self, grad):
        if self.closed:
            return None
        self.dx_calls += 1
        try:
            if self.dx_calls != 1 or tuple(grad.shape) != self.layout.X or grad.dtype != self.torch.bfloat16:
                raise ValueError('duplicate or unsupported native dX')
            self.dx = Slot(grad)
        except (RuntimeError, TypeError, ValueError) as error:
            self.error = error_info(error)
        return None

    def save_dy(self, grad):
        if self.closed:
            return None
        self.dy_calls += 1
        try:
            if self.dy_calls != 1:
                raise ValueError('duplicate native dY')
            self.dy = self.layout.OwnedUpstream(grad, self.torch)
        except (RuntimeError, TypeError, ValueError) as error:
            self.error = error_info(error)
        return None

    def arm_input(self, value):
        alias = value.view_as(value)
        if (alias.data_ptr() != value.data_ptr() or alias.stride() != value.stride()
                or alias.storage_offset() != value.storage_offset() or not alias.requires_grad):
            raise ValueError('native dX view changed input identity')
        self.hooks.append(alias.register_hook(self.save_dx))
        return alias

    def arm_output(self, result):
        if (tuple(result.shape) != self.layout.DY or result.dtype != self.torch.bfloat16
                or not result.requires_grad or result.device != self.x.tensor.device):
            raise ValueError('unsupported native frozen-base output')
        self.y = Slot(result)
        self.hooks.append(result.register_hook(self.save_dy))

    def release(self):
        self.closed = True
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.x = self.weight = self.y = self.dx = self.dy = None


class Observer:
    def __init__(self, torch, layout, trainer, root, base, deadline, emit):
        self.torch, self.layout = torch, layout
        self.trainer, self.root, self.base, self.deadline, self.emit = trainer, root, base, deadline, emit
        self.root_handles = []
        self.micro = None
        self.edge = None
        self.finished = False
        self.installed = False
        self.model_depth = 0
        self.result = dict(schema='qwenvl.native_vproj_upstream_layout_observer.v1', rank=1, site=SITE,
                           eligible_microbatch_limit=MAX_MICROBATCHES, diagnostics_invoked=0, coverage=[], state='ACTIVE',
                           diagnostic_optimizer_updates=0, formal_training_qualified=False)
        self.original_step, self.original_forward = trainer.training_step, base.forward
        self.step_wrapper, self.forward_wrapper = self.training_step, self.leaf
        self.step_attr = ('training_step' in vars(trainer), vars(trainer).get('training_step'))
        self.forward_attr = ('forward' in vars(base), vars(base).get('forward'))

    def install(self):
        try:
            self.root_handles.append(self.root.register_forward_pre_hook(self.enter_root))
            self.root_handles.append(self.root.register_forward_hook(self.leave_root, always_call=True))
            self.trainer.training_step = self.step_wrapper
            self.base.forward = self.forward_wrapper
            self.installed = True
        except Exception:
            self.uninstall()
            raise

    @staticmethod
    def restore(owner, name, wrapper, saved):
        if getattr(owner, name) is wrapper:
            if saved[0]:
                setattr(owner, name, saved[1])
            else:
                delattr(owner, name)

    def uninstall(self):
        self.restore(self.trainer, 'training_step', self.step_wrapper, self.step_attr)
        self.restore(self.base, 'forward', self.forward_wrapper, self.forward_attr)
        for handle in self.root_handles:
            handle.remove()
        self.root_handles.clear()
        if self.edge:
            self.edge.release()
            self.edge = None
        self.installed = False

    def finish(self, state):
        if self.finished:
            return
        self.finished = True
        self.result['state'] = state
        self.uninstall()
        try:
            self.emit(self.result)
        except (OSError, TypeError, ValueError) as error:
            self.result['receipt_error'] = error_info(error)

    def enter_root(self, _module, _inputs):
        if self.micro is not None:
            self.micro['root_forwards'] += 1
            self.model_depth += 1

    def leave_root(self, _module, _inputs, _output):
        if self.micro is not None:
            self.model_depth = max(0, self.model_depth-1)

    def select(self, value):
        torch = self.torch
        return (isinstance(value, torch.Tensor) and torch.is_grad_enabled()
                and self.root.training and self.base.training and value.requires_grad
                and value.is_cuda and value.dtype == torch.bfloat16 and tuple(value.shape) == self.layout.X
                and value.is_contiguous() and self.base.bias is None
                and tuple(self.base.weight.shape) == self.layout.W and self.base.weight.is_contiguous()
                and not self.base.weight.requires_grad and self.base.weight.dtype == torch.bfloat16
                and self.base.weight.device == value.device and value.device.index == torch.cuda.current_device() == 1)

    def leaf(self, *args, **kwargs):
        micro = self.micro
        if micro is None:
            return self.original_forward(*args, **kwargs)
        if self.model_depth == 0:
            micro['leaf_outside_root_calls'] += 1
            return self.original_forward(*args, **kwargs)
        micro['leaf_calls'] += 1
        positional = len(args) == 1 and not kwargs
        keyword = not args and set(kwargs) == {'input'}
        if not (positional or keyword):
            micro['unsupported_leaf_calls'] += 1
            return self.original_forward(*args, **kwargs)
        value = args[0] if positional else kwargs['input']
        if isinstance(value, self.torch.Tensor):
            shape = dict(shape=list(value.shape), dtype=str(value.dtype), requires_grad=bool(value.requires_grad))
            if shape not in micro['leaf_shapes'] and len(micro['leaf_shapes']) < 8:
                micro['leaf_shapes'].append(shape)
        if not self.select(value):
            micro['unmatched_leaf_calls'] += 1
            return self.original_forward(*args, **kwargs)
        micro['eligible_leaf_calls'] += 1
        if micro['root_forwards'] != 1 or self.model_depth != 1 or self.edge is not None or time.time() >= self.deadline:
            return self.original_forward(*args, **kwargs)
        try:
            self.edge = Edge(self.torch, self.layout, value, self.base.weight)
            alias = self.edge.arm_input(value)
        except (RuntimeError, TypeError, ValueError) as error:
            micro['attach_error'] = error_info(error)
            if self.edge:
                self.edge.release()
                self.edge = None
            return self.original_forward(*args, **kwargs)
        # Invoke the real leaf exactly once. Its exception belongs to the native
        # step, and an observer failure below must never retry this call.
        output = self.original_forward(alias) if positional else self.original_forward(input=alias)
        try:
            self.edge.arm_output(output)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            self.edge.error = error_info(error)
        return output

    def observe_after_backward(self, micro):
        edge = self.edge
        torch = self.torch
        micro.update(native_dx_hooks=edge.dx_calls, native_dy_hooks=edge.dy_calls)
        if (edge.error or edge.dy is None or edge.dx is None or edge.y is None
                or micro['root_forwards'] != 1 or micro['eligible_leaf_calls'] != 1):
            micro['diagnostic'] = dict(candidate_status='UNSUPPORTED_NATIVE_EDGE', planned_measured_trials=0,
                                       completed_measured_trials=0, failure=edge.error)
            return
        device = edge.x.tensor.device
        cutoff = time.monotonic() + MAX_DIAGNOSTIC_SECONDS

        def check():
            if time.time() >= self.deadline or time.monotonic() >= cutoff:
                raise TimeoutError('bounded post-backward diagnostic deadline')

        try:
            check()
            with torch.random.fork_rng(devices=[device.index], enabled=True):
                self.result['diagnostics_invoked'] += 1
                outcome = self.layout.evaluate(torch, torch.nn.functional.linear, edge.x.get(), edge.weight.get(),
                                               edge.dy, edge.y.get(), edge.dx.get(), check)
            micro['diagnostic'] = outcome
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
            micro['diagnostic'] = dict(candidate_status='OBSERVATION_REJECTED', failure=error_info(error))

    def training_step(self, model, *args, **kwargs):
        if self.finished or self.micro is not None:
            return self.original_step(model, *args, **kwargs)
        if time.time() >= self.deadline:
            self.finish('DEADLINE_BEFORE_NATIVE_STEP')
            return self.original_step(model, *args, **kwargs)
        state = self.trainer.state
        micro = dict(ordinal=len(self.result['coverage'])+1, global_step=state.global_step, root_forwards=0,
                     leaf_calls=0, leaf_outside_root_calls=0, unmatched_leaf_calls=0, unsupported_leaf_calls=0,
                     eligible_leaf_calls=0, leaf_shapes=[], native_step_returned=False)
        self.result['coverage'].append(micro)
        self.micro = micro
        try:
            loss = self.original_step(model, *args, **kwargs)
        except BaseException:
            self.finish('NATIVE_STEP_RAISED')
            raise
        finally:
            self.micro = None
            self.model_depth = 0
        micro['native_step_returned'] = True
        if self.edge:
            self.observe_after_backward(micro)
            self.finish('ONE_NATIVE_EDGE_ATTEMPTED')
        elif micro['ordinal'] == MAX_MICROBATCHES:
            self.finish('WINDOW_NO_ELIGIBLE_EDGE')
        return loss


def validate_deadline(deadline):
    if not math.isfinite(deadline) or time.time() >= deadline:
        raise ValueError('missing future absolute observation deadline')
    return deadline
