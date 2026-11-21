"""Compile and execute the exact rendered method with instrumented local providers."""

from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent


def main():
    module = runpy.run_path(str(HERE / 'render.py'))
    sources, declaration, symbol = (module[name] for name in ('sources', 'DECL', 'SYMBOL'))
    original = module['PARENT'] / 'deepseek_v4_attention_prepare.cpp'
    raw, header = original.read_bytes(), original.with_suffix('.hpp').read_bytes()
    bundle = sources(raw, header)
    caller = bundle['caller'].decode().replace('\r\n', '\n')
    baseline = bundle['baseline_caller'].decode().replace('\r\n', '\n')
    stripped = caller.replace('\n' + declaration.format(name=symbol), '').replace(
        'arm == inverse_output::Arm::h1_pack32 ? ' + symbol + ' : (', '').replace(
        'deep_inverse_rope_h2_v1);', 'deep_inverse_rope_h2_v1;')
    assert stripped == baseline
    assert bundle['caller'].count(symbol.encode()) == 2
    assert bundle['policy'].replace(b', h1_pack32', b'').replace(
        b'    if (std::string_view(value) == "h1-pack32") {\n        return Arm::h1_pack32;\n    }\n', b'') == bundle['baseline_policy']
    try:
        sources(raw + b'\n', header)
    except ValueError:
        pass
    else:
        raise AssertionError('changed archived input accepted')

    start = caller.index('rt::Tensor AttentionPrepare::inverse_output_rope(')
    end = caller.index('\n}\n}', start) + len('\n}')
    rendering = caller[start:end]
    compiler = shutil.which('g++')
    if compiler is None:
        raise RuntimeError('local C++ compiler unavailable')
    cases = [
        ('unset_legacy', None, 'normal', 'ACLY', ''),
        ('explicit_legacy', 'legacy', 'normal', 'ACLY', ''),
        ('original_h1', 'h1', 'normal', 'ASTU1Y', ''),
        ('original_h2', 'h2', 'normal', 'ASTU2Y', ''),
        ('pack32', 'h1-pack32', 'normal', 'ASTUPY', ''),
        ('multi_token_fallback', 'h1-pack32', 'multi', 'ACLY', ''),
        ('head_count_fallback', 'h1-pack32', 'other-heads', 'ACLY', ''),
        ('noncontiguous_fallback', 'h1-pack32', 'noncontiguous', 'ACLY', ''),
        ('missing_stream_fallback', 'h1-pack32', 'no-stream', 'ASCLY', ''),
        ('invalid_policy', 'pack32', 'normal', 'A', 'Invalid DeepSeek inverse RoPE policy'),
        ('incorrect_dtype', 'h1-pack32', 'bad-dtype', 'A', 'DeepSeek inverse rotary output geometry differs'),
        ('graph_rejection', 'h1-pack32', 'graph', 'A', 'DeepSeek inverse rotary output geometry differs'),
        ('error_no_legacy_replay', 'h1-pack32', 'provider-failure', 'ASTUP', 'DeepSeek fused inverse RoPE launch failed: 31'),
    ]
    with TemporaryDirectory(prefix='deep-pack32-caller-') as temp:
        directory = Path(temp)
        (directory / 'rendered_method.inc').write_bytes(rendering.encode())
        (directory / 'inverse_output_policy.hpp').write_bytes(bundle['policy'])
        executable = directory / 'routes.exe'
        subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pedantic', '-O2',
                        '-I', str(directory), str(HERE / 'check_caller.cpp'), '-o', str(executable)],
                       timeout=30, check=True)
        for _, arm, scenario, expected_trace, error in cases:
            env = os.environ.copy()
            key = 'INFINILM_DEEP_INVERSE_OUTPUT_ROPE'
            if arm is None:
                env.pop(key, None)
            else:
                env[key] = arm
            subprocess.run([str(executable), scenario, expected_trace, error], env=env, timeout=3, check=True)
    print(json.dumps(dict(state='LOCAL_RENDERED_CALLER_ROUTES_PASS', local_host_compiles=1,
                          passed_cases=[case[0] for case in cases], original_bytes_drift_rejected=True,
                          only_provider_declaration_dispatch_and_policy_token_changed=True,
                          rendered_caller_SHA256=sha256(bundle['caller']).hexdigest(),
                          rendered_policy_SHA256=sha256(bundle['policy']).hexdigest(),
                          pack32_target=bundle['pack32_target'], actual_target_compiles=0,
                          actual_Slurm_submissions=0, actual_GPU_trials=0, actual_model_runs=0)))


if __name__ == '__main__':
    main()
