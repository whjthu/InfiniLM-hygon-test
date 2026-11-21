"""Bind the fresh K/V candidate to true decode in the CPU308 Native/Python caller."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'fresh-kv-pack-caller-source-v1'
ORIGINAL_MODEL = HERE.parent / 'full-model-depth-stratified-observer-v1'
BASE_PLAN_SHA = '0bc06dd9b4a9cd2052ea84666bfb8646c2e338e3428896539a1de295d04a7125'
BASE_RENDER_SHA = '201568fb9afa5cd9cf522ccc0895aae57f9bb7505b1afc2dd85d1c5e1cb295cb'
BRIDGE_SHA = '188af7d84bbef358818a6e0a56910270e8e1191824466fd0ac6782525f9c8430'
OWNER_SHA = '31c0a2a22fcf3acdd409e70344a07d3fdb6684bcd3f459b04e50a6a825e50428'
BINDING_SHA = 'ea014b57f1ac1b450ca5acc8cb28e41283d041b80f82e88141377fb005b86fb7'
ARM_SHA = '4d35ae30ee6b03acc2968493bb2ec512f75d2ade7713d2a7bd7fa9e53098e26b'
SYMBOL = 'modern_glm_global_fresh_kv_decode_r474'


def digest(raw):
    return sha256(raw).hexdigest()


def read(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError('frozen bytes differ: ' + path.name)
    return raw


def once(raw, before, after):
    if raw.count(before) != 1:
        raise ValueError('frozen edit anchor differs')
    return raw.replace(before, after)


def sources():
    base = read(BASE / 'render.py', BASE_RENDER_SHA)
    scope = {'__name__': 'code_only', '__file__': str(BASE / 'render.py')}
    exec(compile(base, str(BASE / 'render.py'), 'exec'), scope)
    files = scope['sources']()
    plan = json.loads(read(BASE / 'PLAN.json', BASE_PLAN_SHA))
    if {name: digest(raw) for name, raw in files.items()} != plan['generated_sources_SHA256']:
        raise ValueError('original fresh K/V caller inputs changed')
    cpu = json.loads(read(scope['PARENT'] / 'PLAN.json', scope['PINNED_PLAN']))
    if cpu['units'][0]['name'] != 'u00' or cpu['units'][0]['source_SHA256'] != BRIDGE_SHA:
        raise ValueError('original u00 forward identity differs')
    files['full_bridge_observer.cpp'] = read(scope['PARENT'] / cpu['source_blobs'][BRIDGE_SHA], BRIDGE_SHA)
    files['full_bridge_observer.cpp'] += b'\n#include "forward_explicit_ar.inc"\n'
    include = b'#include "fresh_kv_route.hpp"\n'
    attention = once(files['NoV_tp16.cpp'], include, include + b'#include "fresh_kv_explicit_ar.hpp"\n')
    shape = b'&r, requests.size()==1 && tokens==1'
    files['NoV_tp16.cpp'] = once(attention, shape, b'&r, ::glm_r474_fresh_kv_ar::active_for(&r) && requests.size()==1 && tokens==1')
    for name in ('forward_explicit_ar.inc', 'fresh_kv_explicit_ar.hpp'):
        files[name] = (HERE / name).read_bytes()

    binding = read(ORIGINAL_MODEL / 'typed_binding.py', BINDING_SHA)
    insertion = (
        b'    decode=getattr(library,\'' + SYMBOL.encode() + b'\')\n'
        b"    decode.argtypes,decode.restype=SIGNATURES['forward']\n"
        b"    result['decode_forward']=decode\n"
    )
    binding = once(binding, b'    return result\n', insertion + b'    return result\n')
    binding = once(binding, b'def forward(self,tokens,past,input_pointer,position_pointer,logits_pointer,stream_pointer):',
                   b'def forward(self,tokens,past,input_pointer,position_pointer,logits_pointer,stream_pointer,*,phase="prefill"):')
    binding = once(binding, b"        self.check(self.f['forward'](self.ctx,tokens,1,packet,",
                   b"        if phase not in ('prefill','decode') or (phase=='decode' and (tokens!=1 or past==0)):\n"
                   b"            raise ValueError('expected explicit prefill or a single decode continuation')\n"
                   b"        forward=self.f['decode_forward' if phase=='decode' else 'forward']\n"
                   b"        self.check(forward(self.ctx,tokens,1,packet,")
    files['typed_binding.py'] = binding
    arm = read(ORIGINAL_MODEL / 'native_arm.py', ARM_SHA)
    files['native_arm.py'] = once(arm, BINDING_SHA.encode(), digest(binding).encode())
    owner = read(ORIGINAL_MODEL / 'owner.py', OWNER_SHA)
    call = b'ctx.forward(len(ids),past,tokens.data_ptr(),positions.data_ptr(),logits.data_ptr(),int(stream.cuda_stream))'
    files['owner.py'] = once(owner, call, call[:-1] + b',phase=kind)')
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    files = sources()
    args.output.mkdir()
    for name, raw in files.items():
        (args.output / name).write_bytes(raw)
    result = dict(files_SHA256={name: digest(raw) for name, raw in files.items()},
                  host_units=['u00', 'u15', 'u17'], additional_export=SYMBOL,
                  actual_target_builds=0, actual_GPU_steps=0)
    (args.output / 'BINDING.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
