"""Stage exactly one bounded gfx936 compilation and standalone comparison DSO link."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LABEL = 'GLM474-fresh-kv-headshare2-gfx936-CPU-v1'
TEMPLATE = Path('code/bw1000/Deep-C4-causal-consumer-r483/inverse-output-rope-CPU-v1/compile_once.py')
TEMPLATE_SHA = '763c53389ab8cfddb215c8a5f3dd53b44821179e7e636c95065854b1f438c182'
OLD_PLAN = Path('code/bw1000/GLM-C1-layer-observer-r474/fresh-kv-pack-CPU-v1/PLAN.json')
OLD_PLAN_SHA = 'c94c39997a8986c950dd120c346fa599552b4eeed73ac5b83451bb6e06c3b951'
CUDA_PROVIDER = '/opt/dtk/cuda/cuda/lib64/libcudart.so.12'
CUDA_SHA = 'ac547071522961b136d7000d2268b8cb470b1f085007b3be3cb9039d50281f2e'
EXPORTS = ('glm_r474_fresh_kv_headshare_h1', 'glm_r474_fresh_kv_headshare_h2')


def digest(raw):
    return sha256(raw).hexdigest()


def pinned(path, sha, max_bytes):
    with path.open('rb') as stream:
        raw = stream.read(max_bytes+1)
    if len(raw) > max_bytes or digest(raw) != sha:
        raise ValueError('changed original toolchain source: ' + path.name)
    return raw


def render(raw):
    if digest(raw) != TEMPLATE_SHA:
        raise ValueError('changed original bounded compiler template')
    options = (
        (b'Deep-inverse-output-rope-gfx936-CPU-v1', LABEL.encode(), 1),
        (b'deep_inverse_rope_h1_v1', EXPORTS[0].encode(), 1),
        (b'deep_inverse_rope_h2_v1', EXPORTS[1].encode(), 1),
        (b'"deep_inverse_rope_"', b'"glm_r474_fresh_kv_headshare_"', 1),
        (b'inverse_output_rope.', b'fresh_kv_headshare.', 5),
    )
    for old, new, count in options:
        if raw.count(old) != count:
            raise ValueError('unexpected frozen compiler template boundary')
        raw = raw.replace(old, new)
    at = b'        report.update(passed=True, object=str(obj),'
    addition = f'''        provider = Path({CUDA_PROVIDER!r})
        require(pin(provider) == {CUDA_SHA!r}, "CUDA_PROVIDER_IDENTITY")
        shared = output / "libGLM_fresh_kv_headshare.so"
        command("link", ["/usr/bin/g++", "-shared", str(obj), str(provider), "-Wl,-z,defs",
                         "-Wl,-rpath," + str(provider.parent), "-o", str(shared)], 35)
        report["actual_native_links"] = 1
        dynamic = command("dynamic", ["/usr/bin/readelf", "-d", str(shared)], 10).read_text()
        needed = re.findall(r"Shared library: \\[([^\\]]+)\\]", dynamic)
        require(needed.count(provider.name) == 1 and str(provider.parent) in dynamic,
                "CUDA_PROVIDER_EDGE")
        dynsym = command("exports", ["/usr/bin/nm", "-D", "--defined-only", str(shared)], 10).read_text()
        actual_exports = {{row[-1] for line in dynsym.splitlines() if len(row := line.split()) >= 3
                   and row[-2] == "T" and row[-1].startswith("glm_r474_fresh_kv_headshare_")}}
        require(actual_exports == EXPORTS, "EXACT_PAIR_DSO_EXPORTS")
        report.update(shared=str(shared), shared_SHA256=pin(shared), shared_bytes=shared.stat().st_size,
                      cuda_provider_SHA256=pin(provider), direct_needed=needed)
'''.encode()
    if raw.count(at) != 1:
        raise ValueError('unexpected compile invoker link boundary')
    return raw.replace(at, addition+at)


def build_inputs():
    original = json.loads(pinned(REPO/OLD_PLAN, OLD_PLAN_SHA, 64*1024))
    invoker = render(pinned(REPO/TEMPLATE, TEMPLATE_SHA, 64*1024))
    leaf = (HERE/'fresh_kv_headshare.cu').read_bytes()
    plan = {key: original[key] for key in ('compiler_prefix', 'tools', 'producer_receipt', 'producer_bytes',
                                           'producer_SHA256', 'cuda_runtime_header', 'cuda_runtime_header_SHA256')}
    plan.update(label=LABEL, source_SHA256=digest(leaf), compile_entry_SHA256=digest(invoker),
                geometry=dict(tokens=512, local_heads=4, input_width=448, output_width=256,
                              rotary_width=64, block_threads=128, heads_per_block=[1,2]),
                exports=list(EXPORTS), planned_device_TUs=1, planned_native_links=1, actual_GPU_steps=0)
    sources = {'fresh_kv_headshare.cu':leaf, 'compile_once.py':invoker,
               'PLAN.json':(json.dumps(plan, sort_keys=True,indent=2)+'\n').encode()}
    manifest = (json.dumps({'files_SHA256':{name:digest(raw) for name,raw in sources.items()}},
                           sort_keys=True,indent=2)+'\n').encode()
    return {**sources,'SOURCE_FILES.json':manifest}


def stage(path):
    sources = build_inputs()
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    with TemporaryDirectory(prefix='glm-headshare-stage-', dir=path.parent) as temp:
        tree = Path(temp) / 'source'
        tree.mkdir(mode=0o700)
        for name, raw in sources.items():
            (tree/name).write_bytes(raw)
        tree.rename(path)
    return dict(compile_manifest_SHA256=digest(sources['SOURCE_FILES.json']), planned_device_TUs=1,
                planned_native_links=1, actual_GPU_steps=0, actual_target_compiles=0)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    print(json.dumps(stage(parser.parse_args().output),sort_keys=True))
