"""Optional gradient buffering for a local, unsynchronized DDP microbatch."""


def valid_local_grad(gradient, parameter):
    return (gradient.dtype == parameter.dtype and gradient.device == parameter.device
            and gradient.shape == parameter.shape and gradient.is_contiguous()
            and not gradient.is_sparse and not gradient.requires_grad)


def grouped_local_backward(torch, model, parameters, original):
    """Run the existing training step once, and merge its local FP32 contributions.

    The caller keeps the first accumulation microbatch and the synchronized
    microbatch on the original path. Failure invalidates the current update.
    """
    if (not isinstance(model, torch.nn.parallel.DistributedDataParallel) or model.require_backward_grad_sync
            or model.gradient_as_bucket_view):
        raise ValueError("Expected unsynchronized non-bucket-view DDP")
    saved = [parameter.grad for parameter in parameters]
    if any(not parameter.requires_grad or parameter.dtype != torch.float32
           or (previous is not None and not valid_local_grad(previous, parameter))
           for parameter, previous in zip(parameters, saved, strict=True)):
        raise ValueError("Unsupported native FP32 accumulation input")
    if len(parameters) != len(set(map(id, parameters))) or not parameters:
        raise ValueError("Unexpected trainable parameter identity set")
    if len({parameter.device for parameter in parameters}) != 1:
        raise ValueError("Grouped addition must stay on one local device")
    for parameter in parameters:
        parameter.grad = None
    try:
        result = original()
        additions, contributions = [], []
        for parameter, previous in zip(parameters, saved, strict=True):
            current = parameter.grad
            if current is not None and not valid_local_grad(current, parameter):
                raise ValueError("Unsupported native local gradient contribution")
            if previous is not None and current is not None:
                if previous.untyped_storage().data_ptr() == current.untyped_storage().data_ptr():
                    raise ValueError("Local and retained gradients share storage")
                additions.append(previous)
                contributions.append(current)
        with torch.no_grad():
            for start in range(0, len(additions), 64):
                torch._foreach_add_(additions[start:start+64], contributions[start:start+64])
            for parameter, previous in zip(parameters, saved, strict=True):
                if previous is not None:
                    parameter.grad = previous
    except BaseException:
        for parameter, previous in zip(parameters, saved, strict=True):
            parameter.grad = previous
        raise
    return result, len(additions)
