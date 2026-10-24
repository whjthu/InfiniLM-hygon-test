"""Bounded CPU-only source/ELF verification. Never dlopen, import torch, or compile."""
import argparse
import hashlib
import json
import pathlib
import subprocess

DSO = '0e0512551483457af01944e9c9298442f499d959836a1c5b66dfa438a3c6c4ea'


def verify(root, manifest, local=False):
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    m = json.loads(manifest.read_text())
    incoming = root if local else root / 'incoming'
    for name, digest in m['sources_SHA256'].items():
        assert sha(incoming / name) == digest, ('source_changed', name)
    if local:
        return {'CPU_source_only': True, 'files': len(m['sources_SHA256']), 'native_ELF_symbols_checked': False}
    bp = root / 'results/deep-tailSUM-r7212-build.json'
    assert sha(bp) == 'fc7065a9b56255bb710dbedb53a1b71e6c017f7bdc23bc2a86a41664789a3b0b'
    build = json.loads(bp.read_text())
    lib = root / 'candidates/deep-tailSUM-r7212/libdeep_tailSUM_full_r7212.so'
    assert build['completed'] and sha(lib) == DSO
    def path(value):
        return root / value.removeprefix('/refactor/') if value.startswith('/refactor/') else pathlib.Path(value)
    for name, digest in build['link']['full']['objects'].items():
        assert sha(path(name)) == digest, name
    for name, desc in build['actual_DTK_bindings'].items():
        assert sha(path(name)) == desc['SHA256'], name
    # Compile-input hashes remain in sealed build receipt; no native compile occurs.
    symbols = subprocess.check_output(['nm', '-D', '--defined-only', str(lib)], text=True, timeout=60)
    for name in m['required_symbols']:
        assert any(line.split()[-1] == name for line in symbols.splitlines()), ('missing_symbol', name)
    raw = lib.read_bytes()
    assert b'INFINILM_DEEP_OUTPUT_TILE_R4808\0' in raw
    assert m['geometry']['cache_rows'] == 128 and m['geometry']['cache_capacity'] == 1024
    return {'CPU_source_only': True, 'GPU_initialized': False, 'DSO_SHA256': DSO,
            'native_ELF_symbols_checked': True, 'manifest_SHA256': sha(manifest), 'paths_and_deps_checked': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, default=pathlib.Path('/refactor'))
    parser.add_argument('--manifest', type=pathlib.Path, default=pathlib.Path('/refactor/incoming/Deep-B64-M32-config-r8650-manifest.json'))
    parser.add_argument('--local-source-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.manifest, args.local_source_only)))
