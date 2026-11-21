"""Build one immutable SDXL diagnostic package from the qualified prior lifecycle."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'training-sdxl-frozen-conv-dgrad-fp32-v2'
BASE_PIN = '5584f33e1d977ce7269b2a4461399f2dacb0772174cdd87a012a7c927a264bb1'
LABEL = 'training-sdxl-sdpa-adjoint-fp32-v1'
REQUEST = 'a93d41a0d31140f1ad99bb105a024e99'
REUSED = ('compare_update_CPU.py', 'gencuda_init.py', 'gencuda_runtime.py', 'hip-deps.json',
          'host_device_gate.py', 'hygon_sdxl_assigned_kfd.py', 'hygon_sdxl_compare_first_step.py',
          'hygon_sdxl_first_step.py', 'slurm_accounting.py', 'snapshot.py')


def digest(raw):
    return sha256(raw).hexdigest()


def edit(source, old, new):
    assert source.count(old) == 1, 'frozen template anchor differs'
    return source.replace(old, new)


def render():
    raw = (BASE/'SOURCE_FILES.json').read_bytes()
    assert digest(raw) == BASE_PIN, 'original lifecycle manifest differs'
    inventory = json.loads(raw)['files']

    def pinned(name):
        raw = (BASE/name).read_bytes()
        assert inventory[name] == dict(bytes=len(raw), SHA256=digest(raw)), name
        return raw

    members = {name: pinned(name) for name in REUSED}
    for name in ('adjoint.py', 'native_check.py'):
        members[name] = (HERE/name).read_bytes()
    source = (HERE/'intervention.py').read_text(encoding='utf-8')
    base_path = "Path(__file__).resolve().parent.parent / 'training-sdxl-frozen-conv-dgrad-fp32-v2/hygon_sdxl_first_step.py'"
    members['intervention.py'] = edit(source, base_path, "Path(__file__).with_name('hygon_sdxl_first_step.py')").encode()

    scope = pinned('scope.py').decode()
    members['scope.py'] = edit(scope, "LABEL='training-sdxl-frozen-conv-dgrad-fp32-v2'", f'LABEL={LABEL!r}').encode()

    platform = pinned('run_platform.py').decode()
    start = platform.index('    import intervention_replay\n')
    stop = platform.index('    assert opts==', start)
    platform = platform[:start] + '''    from intervention import replay_once
    output=root/'candidate'
    output.mkdir(mode=0o700)
    evidence=replay_once(Path(reference),Path(policy),Path(weights),output/'replay')
    assert evidence['native_forward_exact'] is True and evidence['actual_adjoint_calls']==140
''' + platform[stop:]
    platform = edit(platform, "    evidence=json.loads((output/'intervention.json').read_bytes())\n", '')
    members['run_platform.py'] = platform.encode()

    campaign = pinned('campaign.py').decode()
    start = campaign.index('    fixture=claim["original_robot_cpu_fixture"]\n')
    stop = campaign.index("    before=snap('pre')", start)
    campaign = campaign[:start] + '''    check=run('-CPU-contract',False,60,container('exec python3 -B -u /t06-run/native_check.py',False),
              cpu_count=4,memory='8192M')
    assert check['passed'], 'target PyTorch CPU contract failed'
    cpu_log=(ROOT/(LABEL+'-CPU-contract.log')).read_bytes()
    assert hashlib.sha256(cpu_log).hexdigest()==check['log_SHA256']
    assert any(json.loads(line)==dict(cpu_fixture_shapes=3,native_dtype='bfloat16',gpu_initialized=False)
               for line in cpu_log.splitlines() if line.startswith(b'{"cpu_fixture_shapes"'))
''' + campaign[stop:]
    campaign = edit(campaign, "    after=snap('post')\n", "    assert gpu['terminal']['terminal_visible'], 'GPU step requires reconciliation'\n    after=snap('post')\n")
    campaign = edit(campaign, "            after=snap('failure-post')\n",
                    "            if is_terminal(one_row(LABEL)):\n                after=snap('failure-post')\n")
    campaign = edit(campaign, "clean_post=after is not None,error_class",
                    "clean_post=bool(after and before and before['pci']==after['pci'] and "
                    "before['boot_id']==after['boot_id']),error_class")
    members['campaign.py'] = campaign.encode()

    claim = json.loads((HERE/'CLAIM.json').read_bytes())
    assert claim['request'] == REQUEST and claim['label'] == LABEL
    members['CLAIM.json'] = (HERE/'CLAIM.json').read_bytes()
    for name, raw in members.items():
        if name.endswith('.py'):
            compile(raw, name, 'exec')
    manifest = {'files': {name: dict(bytes=len(raw), SHA256=digest(raw)) for name, raw in members.items()}}
    members['SOURCE_FILES.json'] = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    return members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    members = render()
    if args.output is not None:
        args.output.mkdir(mode=0o700)
        for name, raw in members.items():
            with (args.output/name).open('xb') as out:
                out.write(raw)
    print(json.dumps(dict(request=REQUEST, target_members=len(members),
                          manifest_SHA256=digest(members['SOURCE_FILES.json']))))


if __name__ == '__main__':
    main()
