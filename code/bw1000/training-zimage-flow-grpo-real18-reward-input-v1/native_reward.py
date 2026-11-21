"""Isolate the exact frozen native CPU reward helper; no RL/model imitation."""
import ast
import hashlib
from pathlib import Path
import tarfile

ARCHIVE = {'bytes':738802,'SHA256':'85cffcb53affa1905b42afec358236089c5e7e094b9ac65d4b464ff39b6f449f'}
MEMBER = 'framework/generation/src/gencuda/backends/zimage_flow_grpo.py'
SOURCE = {'bytes':29853,'SHA256':'b8ad7d7cbab00df4d3df0791937dd5374fca1a13db80a656b83dd5033718734e'}


def need(ok, why):
    if not ok:
        raise ValueError(why)


def pin(raw):
    return {'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest()}


def load_native(archive):
    need(pin(Path(archive).read_bytes()) == ARCHIVE, 'Frozen common framework archive differs')
    with tarfile.open(archive, 'r:gz') as t:
        f = t.extractfile(MEMBER)
        need(f is not None, 'Exact native reward member missing')
        raw = f.read(SOURCE['bytes'] + 1)
    need(pin(raw) == SOURCE, 'Original native reward source differs')
    tree = ast.parse(raw)
    fn, = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'image_prompt_scores']
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    isolated = ast.fix_missing_locations(ast.Module(body=[future,fn],type_ignores=[]))
    ns = {}
    exec(compile(isolated, MEMBER + ':original-image_prompt_scores-only', 'exec'), ns)
    return ns['image_prompt_scores'], {'member':MEMBER,'source':SOURCE,'archive':ARCHIVE,
        'function':'image_prompt_scores','line':fn.lineno,'end_line':fn.end_lineno,
        'function_AST_SHA256':hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest(),
        'whole_backend_or_Torch_imported':False}


def score_batch4(native, images, target_rgb):
    """Input adapter for a future real four-image rollout; no rollout generation."""
    import numpy as np
    need(len(images) == 4, 'Actual RL batch requires all four images; no padding/dropping')
    t = np.asarray(target_rgb, dtype=np.float64)
    need(t.shape == (3,) and np.isfinite(t).all() and (t>=0).all() and (t<=1).all(), 'Invalid frozen target RGB')
    need(len({image.size for image in images}) == 1 and min(images[0].size) >= 2, 'Full common image geometry required')
    values = np.asarray(native(images, tuple(t)), dtype=np.float64)
    need(values.shape == (4,) and np.isfinite(values).all(), 'All four native rewards must be finite')
    return values
