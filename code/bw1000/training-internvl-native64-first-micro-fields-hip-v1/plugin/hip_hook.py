"""Install a stop-before-forward diagnostic at the frozen native hook position."""
from fields import REQUEST, WORLD_SIZE, describe, document, require, valid_sha, write_once


class FirstMicroCaptured(RuntimeError):
    """Intentional stop propagated through the native launcher after all ranks write."""


def install(capture, model, torch, fingerprint, cpu_projection, output):
    """Keep the native rank agreement and make even an exact match stop before forward."""
    rank = capture.rank
    require(type(rank) is int and 0 <= rank < WORLD_SIZE and capture.world_size == WORLD_SIZE
            and len(capture.handles) == 2 and not capture.detached
            and capture.inputs == capture.outputs == capture.updates == capture.clip_calls == 0
            and capture.trainer.state.global_step == 0 and callable(capture.input_agreement)
            and len(capture.input_fingerprints) == 8 and valid_sha(capture.input_fingerprints[0]),
            "Original first-forward capture boundary differs")
    old_pre, old_after = (handle.id for handle in capture.handles)
    old_order = tuple(model._forward_pre_hooks)
    require(old_order and old_order[-1] == old_pre and old_pre in model._forward_pre_hooks_with_kwargs
            and old_after in model._forward_hooks, "Original native pre-hook position differs")
    expected = capture.input_fingerprints[0]
    entered = False

    def pre_forward(_model, args, kwargs):
        nonlocal entered
        if entered:
            raise RuntimeError("Repeated diagnostic first-forward call")
        entered = True
        error = None
        matches = False
        try:
            capture.check_deadline()
            require(capture.trainer.state.global_step == 0 and _model is model, "Wrong native first-forward slot")
            actual, rows = describe(cpu_projection(args), cpu_projection(kwargs), torch,
                                    fingerprint, capture.check_deadline)
            matches = actual == expected
            write_once(output, document("HIP", rank, actual, expected, rows))
        except Exception as failure:
            error = failure
        agreed = capture.input_agreement(error is None and matches)
        if error is not None:
            raise error
        boundary = "MATCHED_INPUT_DIAGNOSTIC_STOP" if agreed else "DIFFERENT_INPUT_DIAGNOSTIC_STOP"
        raise FirstMicroCaptured(f"{REQUEST}:{boundary}")

    # v5 registers its native gate last. The replacement sees the same argv after earlier pre-hooks.
    capture.detach()
    handle = model.register_forward_pre_hook(pre_forward, with_kwargs=True)
    observed = tuple(model._forward_pre_hooks)
    if observed != (*old_order[:-1], handle.id):
        handle.remove()
        raise ValueError("Native pre-hook order changed before diagnostic")
    return handle
