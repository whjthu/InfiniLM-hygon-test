"""Changed stdlib consumers only. No CDLL, Torch, Runtime or Model execution."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import launch_private
import runtime_guard_r442 as guard
import stage_private


def rejected(fn):
    try: fn()
    except (AssertionError, ValueError): return
    raise AssertionError('negative case unexpectedly accepted')


def check_stage(parent, text, config, candidate, output):
    d = stage_private.stage(parent, text, config, candidate, output)
    private_parent, private_text = Path(d['parent']), Path(d['text'])
    changed = []
    for base, private in [(Path(parent), private_parent), (Path(text), private_text)]:
        for name in json.loads((base / 'SOURCE_MANIFEST.json').read_bytes())['files_SHA256']:
            if (base / name).read_bytes() != (private / name).read_bytes(): changed.append(name)
    assert sorted(changed) == sorted(['public_kimi_native/runtime/native_binding_r9143.py',
        'public_kimi_native/owner.py', 'public_kimi_text/cli.py']), changed
    for path in [private_parent / 'public_kimi_native/owner.py',
                 private_parent / 'public_kimi_native/runtime/native_binding_r9143.py',
                 private_text / 'public_kimi_text/cli.py']:
        compile(path.read_bytes(), str(path), 'exec')
    spec = importlib.util.spec_from_file_location('r442_actual_cli', private_text / 'public_kimi_text/cli.py')
    cli = importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    cli.verify_tree(private_parent, d['parent_MF_SHA256'])
    cli.verify_tree(private_text, d['text_MF_SHA256'])
    cfg = json.loads(Path(d['config']).read_bytes())
    roles = []
    for node in range(4):
        c = dict(cfg, run='CPU-r442', nonce='n'*32, node=node, actual_step='910407.CPU',
                 gpu_start_epoch=1, source_manifest_SHA256=d['text_MF_SHA256'])
        env = dict(SLURM_PROCID=str(node), SLURM_JOB_ID='910407', SLURM_STEP_ID='CPU')
        roles.append(cli.configure(c, env, hostname=c['nodes'][node]))
    assert [r['PP_stage'] for r in roles] == [0, 0, 1, 1]
    assert all(r['global_cards'] == 32 for r in roles)
    alias = Path(d['runtime_directory']) / 'libinfinilm_runtime.so'
    manifest = json.loads(Path(cfg['model_manifest']).read_bytes())
    maps = '0000-1000 r-xp 0 00:01 42 ' + str(alias.resolve()) + '\n'
    probe_calls = []
    def probe(p):
        probe_calls.append(str(p))
        return {name: str(alias.resolve()) for name in guard.SYMBOLS}
    guard.verify_runtime(manifest, maps, probe)
    rejected(lambda: guard.verify_runtime(manifest, maps + '1000-2000 r-xp 0 00:01 43 /old/libinfinilm_runtime.so\n', probe))
    rejected(lambda: guard.verify_runtime(manifest, maps.replace(str(alias.resolve()), '/old/libinfinilm_runtime.so'), probe))
    rejected(lambda: guard.verify_runtime(manifest, maps.rstrip()+' (deleted)\n', probe))
    rejected(lambda: guard.verify_runtime(manifest, maps, lambda p:{name:'/old/libinfinilm_runtime.so' for name in guard.SYMBOLS}))
    assert guard.checked_load_delta([0,2], [123,9])['chosen'] == 123
    rejected(lambda: guard.checked_load_delta([0,2], [0,9]))
    rejected(lambda: guard.checked_load_delta([5,2], [4,9]))
    # Default-OFF deployment is never runnable. Only construct the future command
    # from a separate synthetic authorization copy; execve is NOT called.
    deployment = Path(output) / 'DEPLOYMENT.json'
    rejected(lambda: launch_private.command(deployment, stage_private.sha(deployment), [], {}))
    test_cfg = dict(cfg, execute_authorized=True)
    command_cfg = Path(output) / 'CPU-authorized-command-only.json'
    stage_private.write_json(command_cfg, test_cfg)
    td = dict(d, config=str(command_cfg), config_SHA256=stage_private.sha(command_cfg))
    command_proof = Path(output) / 'CPU-command-proof.json'
    stage_private.write_json(command_proof, td)
    args = ['--node', '0', '--run', 'CPU-only', '--serve']
    env = {'LD_LIBRARY_PATH':'/qualified/sdk', 'LD_PRELOAD':'/qualified/libblas_policy.so',
           'PYTHONPATH':'/original/python', 'INFINILM_KIMI_MOE_EXECUTION_R4665':'0'}
    cmd, e = launch_private.command(command_proof, stage_private.sha(command_proof), args, env)
    assert e['LD_LIBRARY_PATH'] == d['runtime_directory'] + ':/qualified/sdk'
    assert e['LD_PRELOAD'] == env['LD_PRELOAD'] and e['INFINILM_KIMI_MOE_EXECUTION_R4665'] == '0'
    assert e['INFINILM_CPU_ROW_COPY_R436'] == '1'
    assert cmd[-len(args):] == args
    rejected(lambda: launch_private.command(command_proof, stage_private.sha(command_proof), ['--source-sha=bad'], env))
    rejected(lambda: launch_private.command(command_proof, stage_private.sha(command_proof), args, dict(env, INFINILM_CPU_ROW_COPY_R436='0')))
    rejected(lambda: stage_private.stage(parent, text, config, candidate, output))
    return dict(passed=True, actual_source_configure_roles=roles, changed_original_members=changed,
        default_OFF=True, sole_provider_maps_and_symbol_path_negative_cases=4,
        real_weight_load_counter_delta_cases=3, launch_command_only=True,
        Runtime_executed=False, Native_dlopen=False, GPU=False, Model_READY=False,
        proof=d, symbol_probe_leaf='recording only; actual dladdr remains device-prefactory requirement')


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser()
    for name in ['parent','text','config','candidate','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(check_stage(a.parent,a.text,a.config,a.candidate,a.output),indent=2))
