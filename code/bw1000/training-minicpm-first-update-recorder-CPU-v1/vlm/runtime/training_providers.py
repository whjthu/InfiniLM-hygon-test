"""Record bound model callables and already loaded modules without importing kernels."""
import hashlib
import inspect
from pathlib import Path
import sys


def provider_inventory(model, *, model_module_prefix, expected_model_source_SHA256):
    """Bindings are evidence of selection, not a trace of executed GPU kernels."""
    cache = {}

    def file_pin(filename):
        path = Path(filename).resolve(strict=True)
        if path not in cache:
            size = path.stat().st_size
            if not 0 < size <= 512 * 1024**2:
                raise ValueError('provider file outside bounded size')
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for raw in iter(lambda: stream.read(1024**2), b''):
                    h.update(raw)
            cache[path] = dict(path=str(path), bytes=size, SHA256=h.hexdigest())
        return cache[path]

    def callable_pin(value):
        if value is None:
            return dict(selected=False)
        if not callable(value):
            raise ValueError('selected provider is not callable')
        wrapper = type(value).__module__ + '.' + type(value).__qualname__
        function = getattr(value, '__func__', value)
        # Triton JIT wrappers expose the Python function as fn. Do not execute it.
        if inspect.isfunction(getattr(function, 'fn', None)):
            function = function.fn
        function = inspect.unwrap(function)
        module = sys.modules.get(getattr(function, '__module__', ''))
        filename = inspect.getsourcefile(function) or getattr(module, '__file__', None)
        if filename is None:
            raise ValueError('selected provider has no source or native module file')
        return dict(selected=True, module=function.__module__,
                    qualname=getattr(function, '__qualname__', type(function).__qualname__),
                    wrapper=wrapper, first_line=getattr(getattr(function, '__code__', None), 'co_firstlineno', None),
                    file=file_pin(filename))

    bindings = []
    attributes = ('causal_conv1d_fn', 'causal_conv1d_update',
                  'chunk_gated_delta_rule', 'recurrent_gated_delta_rule')
    for name, module in model.named_modules():
        cls = type(module)
        if not cls.__module__.startswith(model_module_prefix):
            continue
        if not all(hasattr(module, attr) for attr in attributes):
            continue
        source = file_pin(inspect.getfile(cls))
        if source['SHA256'] != expected_model_source_SHA256:
            raise ValueError('actual model provider source differs from frozen source')
        calls = {attr: callable_pin(getattr(module, attr)) for attr in attributes}
        if any(not calls[attr]['selected'] for attr in attributes[1:]):
            raise ValueError('required recurrent/chunk provider absent')
        bindings.append(dict(name=name, type=cls.__module__ + '.' + cls.__qualname__,
                             source=source, callables=calls))
    if not bindings:
        raise ValueError('no actual gated-delta model provider bindings')
    loaded = {}
    for name in ('torch', 'torch._C', 'triton', 'triton._C.libtriton',
                 'causal_conv1d_cuda', 'fla.ops.gated_delta_rule'):
        module = sys.modules.get(name)
        if module is None:
            loaded[name] = dict(loaded=False)
        elif getattr(module, '__file__', None) is None:
            loaded[name] = dict(loaded=True, file=None)
        else:
            loaded[name] = dict(loaded=True, file=file_pin(module.__file__))
    return dict(bindings=bindings, already_loaded_modules=loaded,
                qualification='Bound Python callables and loaded module files only; no kernel-call trace or complete ELF dependency closure.')
