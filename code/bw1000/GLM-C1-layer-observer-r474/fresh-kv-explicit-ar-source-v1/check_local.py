"""Exercise the generated caller and actual C ABI wrapper on host only."""

import argparse
import ast
import json
from pathlib import Path
import runpy
import subprocess
import tempfile
import threading


HERE = Path(__file__).resolve().parent


class Function:
    def __init__(self, key, calls):
        self.key = key
        self.calls = calls
        self.rc = 0

    def __call__(self, *args):
        if self.key == 'error':
            return b'boundary fixture failed'
        ctx, tokens, count, packet, inputs, positions, logits, all_positions, stream = args
        assert (ctx, count, inputs.value, positions.value, logits.value, all_positions, stream.value) == (1, 1, 7, 8, 9, 0, 10)
        self.calls.append((self.key, tokens, tuple(packet)))
        return self.rc


def check_caller(files, symbol):
    assert all(ast.parse(files[name]) for name in ('owner.py', 'typed_binding.py', 'native_arm.py'))
    owner = ast.parse(files['owner.py'])
    assert sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and isinstance(node.func.value, ast.Name) and node.func.value.id == 'ctx'
               and node.func.attr == 'forward'
               and any(kw.arg == 'phase' and isinstance(kw.value, ast.Name) and kw.value.id == 'kind'
                       for kw in node.keywords) for node in ast.walk(owner)) == 1
    assert {node.args[2].value for node in ast.walk(owner)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'forward'
            and len(node.args) == 3 and isinstance(node.args[2], ast.Constant)} == {'prefill', 'decode'}
    scope = {'__name__': 'local_boundary_check'}
    exec(compile(files['typed_binding.py'], 'typed_binding.py', 'exec'), scope)
    calls = []

    class Library:
        def __getattr__(self, name):
            if name == symbol:
                key = 'decode_forward'
            else:
                key = name.removeprefix('modern_glm_global_').removesuffix('_r470')
            return Function(key, calls)

    ctx = scope['GlobalTP16Context'].__new__(scope['GlobalTP16Context'])
    ctx.thread = threading.get_ident()
    ctx.ready, ctx.poisoned, ctx.ctx, ctx.past = True, False, 1, 0
    ctx.f = scope['attach'](Library())
    assert ctx.f['decode_forward'].argtypes == ctx.f['forward'].argtypes
    assert ctx.f['decode_forward'].restype == ctx.f['forward'].restype
    for tokens, past, kind in [(1, 0, 'decode'), (4, 0, 'decode'), (1, 0, 'other')]:
        try:
            ctx.forward(tokens, past, 7, 8, 9, 10, phase=kind)
        except ValueError:
            assert not calls and ctx.past == 0
        else:
            raise AssertionError('invalid phase reached C ABI')
    ctx.forward(512, 0, 7, 8, 9, 10)
    ctx.forward(1, 512, 7, 8, 9, 10, phase='prefill')
    ctx.forward(1, 513, 7, 8, 9, 10, phase='decode')
    ctx.f['decode_forward'].rc = 1
    try:
        ctx.forward(1, 514, 7, 8, 9, 10, phase='decode')
    except RuntimeError as error:
        assert str(error) == 'boundary fixture failed' and ctx.poisoned and ctx.past == 514
    else:
        raise AssertionError('Native failure was concealed')
    assert calls == [('forward', 512, (0, 0, 512)), ('forward', 1, (0, 512, 1)),
                     ('decode_forward', 1, (0, 513, 1)), ('decode_forward', 1, (0, 514, 1))]
    return calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('compiler')
    args = parser.parse_args()
    render = runpy.run_path(str(HERE / 'render.py'))
    files = render['sources']()
    calls = check_caller(files, render['SYMBOL'])
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        for name in ('forward_explicit_ar.inc', 'fresh_kv_explicit_ar.hpp'):
            (work / name).write_bytes(files[name])
        (work / 'check_bridge.cpp').write_bytes((HERE / 'check_bridge.cpp').read_bytes())
        (work / 'attention.cpp').write_text(
            '#include "fresh_kv_explicit_ar.hpp"\n'
            'bool seen_from_attention(const void* owner) {\n'
            '    return glm_r474_fresh_kv_ar::active_for(owner);\n}\n', encoding='ascii')
        binary = work / 'check_bridge.exe'
        subprocess.run([args.compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                        '-pedantic', '-pthread', 'check_bridge.cpp', 'attention.cpp', '-o', str(binary)],
                       cwd=work, check=True, timeout=30)
        subprocess.run([str(binary)], cwd=work, check=True, timeout=10)
    print(json.dumps({'pass': True, 'python_calls': calls, 'python_rejections_before_dispatch': 3,
                      'native_rejections_before_forward': 9, 'nested_rejections': 3,
                      'cross_thread_checks': 3, 'bridge_fixture_forwards': 4,
                      'generated_sources_SHA256': {name: render['digest'](raw) for name, raw in files.items()},
                      'target_builds': 0, 'native_runtime_loads': 0, 'gpu_steps': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
