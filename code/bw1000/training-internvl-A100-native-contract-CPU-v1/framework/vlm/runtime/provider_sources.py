"""Pin every declared native model family, including embedded text decoders."""
import hashlib
import inspect
from pathlib import Path
import re

from .native_recipe_contract import model_contract


def model_provider_sources(model, expected, case):
    prefix = model_contract(case)['source_prefix']
    prefixes = prefix if isinstance(prefix, tuple) else (prefix,)
    if isinstance(prefix, tuple):
        if not isinstance(expected, dict) or set(expected) != set(prefixes):
            raise ValueError('complete native model-family source pins required')
        pins = expected
    else:
        pins = {prefix: expected}
    if any(not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None
           for value in pins.values()):
        raise ValueError('invalid native model-family source SHA256')
    files, classes, seen = {}, [], set()
    for name, module in model.named_modules():
        cls = type(module)
        family = next((p for p in prefixes if cls.__module__.startswith(p)), None)
        if family is None:
            continue
        path = Path(inspect.getfile(cls)).resolve()
        if str(path) not in files:
            if not 0 < path.stat().st_size <= 2 * 1024**2:
                raise ValueError('native VLM provider outside source bound')
            raw = path.read_bytes()
            files[str(path)] = dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())
        if files[str(path)]['SHA256'] != pins[family]:
            raise ValueError('native VLM implementation source differs')
        classes.append(dict(name=name, type=cls.__module__ + '.' + cls.__qualname__, file=str(path)))
        seen.add(family)
    if seen != set(prefixes):
        raise ValueError('native VLM model-family provider is absent')
    return dict(files=files, classes=classes,
                qualification='Model class/source binding only; no executed kernel trace.')
