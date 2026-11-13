"""Opt-in first-update capture for the pinned HunyuanVideo SFT backend."""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap

BACKEND_SHA256 = '2331d0fa942eb3908dc5392df6008355524f013a0fbdb5067202f9a74400ccae'

# The observer is invoked after each original boundary, except the initial capture.
BOUNDARIES = {
    'optimizer.zero_grad(set_to_none=True)': ('before', 'transformer, optimizer, latents, cond, text_emb, text_mask, byt5_text_states, byt5_text_mask'),
    'hidden_states = torch.cat((latents_noised, cond), dim=1)': ('inputs', 'noise, timestep, hidden_states, target'),
    'loss = torch.nn.functional.mse_loss(prediction, target)': ('forward', 'prediction, loss'),
    'loss.backward()': ('preclip', 'transformer'),
    'grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)': ('postclip', 'transformer, grad_norm'),
    'optimizer.step()': ('updated', 'transformer, optimizer'),
}


def instrument_source(source):
    """Insert observations and prove the underlying function AST is unchanged."""
    original = ast.parse(textwrap.dedent(source))
    tree = ast.parse(textwrap.dedent(source))
    function, = tree.body
    if not isinstance(function, ast.FunctionDef) or function.name != 'run_hunyuanvideo_training' or function.decorator_list:
        raise ValueError('unexpected native HunyuanVideo entry')
    step, = (node for node in ast.walk(function)
             if isinstance(node, ast.FunctionDef) and node.name == 'train_step')
    anchors = {ast.dump(ast.parse(code).body[0]): detail for code, detail in BOUNDARIES.items()}
    seen = []
    inserted = []

    def walk(node):
        for name, values in ast.iter_fields(node):
            if not isinstance(values, list) or not all(isinstance(v, ast.stmt) for v in values):
                continue
            updated = []
            for statement in values:
                anchor = anchors.get(ast.dump(statement))
                updated.append(statement)
                if anchor is not None:
                    phase, args = anchor
                    hook = ast.parse(f'if global_step == 1: _hunyuan_capture.{phase}({args})').body[0]
                    inserted.append((updated, hook))
                    seen.append(phase)
                    updated.append(ast.copy_location(hook, statement))
                walk(statement)
            setattr(node, name, updated)

    walk(step)
    if seen != [phase for phase, _ in BOUNDARIES.values()]:
        raise ValueError('native HunyuanVideo capture boundaries changed')
    code = compile(ast.fix_missing_locations(tree), '<hunyuanvideo-first-update>', 'exec')
    for sequence, hook in inserted:
        sequence.remove(hook)
    if ast.dump(tree) != ast.dump(original):
        raise ValueError('observer changed the original training AST')
    return code


def instrument_backend(original, observer):
    source_path = Path(inspect.getsourcefile(original))
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != BACKEND_SHA256:
        raise ValueError('native walltime backend SHA differs')
    namespace = dict(original.__globals__, _hunyuan_capture=observer)
    exec(instrument_source(inspect.getsource(original)), namespace)
    return namespace[original.__name__]
