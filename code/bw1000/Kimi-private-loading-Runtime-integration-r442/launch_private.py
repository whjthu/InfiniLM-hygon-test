"""Opt-in exec of the unchanged public CLI, before the dynamic loader starts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from runtime_guard_r442 import RUNTIME_SHA

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def command(deployment, expected_sha, argv, environ):
    assert sha(deployment) == expected_sha
    d = json.loads(Path(deployment).read_bytes())
    assert d['Runtime_SHA256'] == RUNTIME_SHA
    cfg = json.loads(Path(d['config']).read_bytes())
    assert sha(d['config']) == d['config_SHA256']
    assert cfg['execute_authorized'] is True, 'Future explicit resource claim required'
    assert cfg['Model_READY'] is False
    assert sha(Path(d['runtime_directory']) / 'libinfinilm_runtime.so') == RUNTIME_SHA
    for key in ['parent', 'text']:
        assert sha(Path(d[key]) / 'SOURCE_MANIFEST.json') == d[key + '_MF_SHA256']
    forbidden = ['--config', '--config-sha', '--source-sha']
    assert not any(v == k or v.startswith(k + '=') for v in argv for k in forbidden)
    assert environ.get('INFINILM_CPU_ROW_COPY_R436', '1') == '1'
    env = dict(environ)
    env['INFINILM_CPU_ROW_COPY_R436'] = '1'  # Runtime caches on first copy; set before exec/import.
    env['LD_LIBRARY_PATH'] = d['runtime_directory'] + ':' + env.get('LD_LIBRARY_PATH', '')
    # Keep the qualified SDK/policy preload and all original environment unchanged.
    env['PYTHONPATH'] = d['text'] + ':' + d['parent'] + ':' + env.get('PYTHONPATH', '')
    cmd = [sys.executable, '-B', '-u', '-m', 'public_kimi_text', '--config', d['config'],
        '--config-sha', d['config_SHA256'], '--source-sha', d['text_MF_SHA256'], *argv]
    return cmd, env


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--enable-private-runtime-r442', action='store_true')
    p.add_argument('--deployment', type=Path, required=True)
    p.add_argument('--deployment-sha', required=True)
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    a = p.parse_args()
    assert a.enable_private_runtime_r442, 'default OFF'
    rest = a.arguments[1:] if a.arguments[:1] == ['--'] else a.arguments
    cmd, env = command(a.deployment, a.deployment_sha, rest, os.environ)
    os.execve(sys.executable, cmd, env)


if __name__ == '__main__': main()
