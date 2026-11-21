"""Add the pack32 arm to the exact existing DeepSeek inverse-output caller."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE.parents[3] / 'code/bw1000/Deep-PD-disaggregation-r382/native/original'
CALLER = HERE.parent / 'inverse-output-rope-caller-v1'
PACK32 = HERE.parent / 'inverse-output-rope-prefix-pack32-v1'
SYMBOL = 'deep_inverse_rope_h1_pack32_v1'
DECL = 'extern "C" int {name}(void*, const void*, const void*, size_t, size_t, void*);'


def checked(path, wanted):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != wanted:
        raise ValueError('frozen source differs: ' + path.name)
    return raw


def replace_once(body, old, new):
    if body.count(old) != 1:
        raise ValueError('frozen caller anchor differs')
    return body.replace(old, new)


def sources(original, original_header):
    boundary = checked(CALLER / 'render_caller.py',
                       '203570d5d7702410fcba3151b77d0138e47e22e22168afb4ad7dd00f6713fef8')
    table = checked(CALLER / 'inverse_output_policy.hpp',
                    '88d232b5c3e2f42c2924373751317e82488f397625384809197ab444458cfc51').decode()
    pack = json.loads(checked(PACK32 / 'LOCAL_RESULT.json',
                              '495721b4add910582d7a57f1d7b9aa5344b55e1a3dfa96f850f3c8d9890476b2'))
    scope = {'__name__': 'frozen_caller'}
    exec(compile(boundary, 'frozen_caller', 'exec'), scope)
    base = scope['render'](original, original_header).decode()
    caller = replace_once(base, DECL.format(name='deep_inverse_rope_h2_v1'),
                          DECL.format(name='deep_inverse_rope_h2_v1') +
                          ('\r\n' if '\r\n' in base else '\n') + DECL.format(name=SYMBOL))
    baseline = 'arm == inverse_output::Arm::h1 ? deep_inverse_rope_h1_v1 : deep_inverse_rope_h2_v1'
    selected = 'arm == inverse_output::Arm::h1_pack32 ? ' + SYMBOL + ' : (' + baseline + ')'
    caller = replace_once(caller, 'const auto launch = ' + baseline + ';', 'const auto launch = ' + selected + ';')
    policy = replace_once(table, 'enum class Arm { legacy, h1, h2 };', 'enum class Arm { legacy, h1, h2, h1_pack32 };')
    old = '    throw std::invalid_argument("Invalid DeepSeek inverse RoPE policy");'
    policy = replace_once(policy, old, '    if (std::string_view(value) == "h1-pack32") {\n'
                          '        return Arm::h1_pack32;\n    }\n' + old)
    return dict(caller=caller.encode(), policy=policy.encode(), baseline_caller=base.encode(),
                baseline_policy=table.encode(), pack32_target=pack['target_package'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    stem = PARENT / 'deepseek_v4_attention_prepare.cpp'
    bundle = sources(stem.read_bytes(), stem.with_suffix('.hpp').read_bytes())
    args.output.mkdir()
    names = {'deepseek_v4_attention_prepare.cpp': 'caller', 'inverse_output_policy.hpp': 'policy'}
    files = {}
    for name, field in names.items():
        raw = bundle[field]
        (args.output / name).write_bytes(raw)
        files[name] = sha256(raw).hexdigest()
    handoff = dict(files_SHA256=files, pack32_candidate=bundle['pack32_target'], pack32_symbol=SYMBOL,
                   linked_candidate_provider_verified=False, actual_target_builds=0, device_qualified=False)
    (args.output / 'CALLER_BINDING.json').write_text(json.dumps(handoff, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(handoff))


if __name__ == '__main__':
    main()
