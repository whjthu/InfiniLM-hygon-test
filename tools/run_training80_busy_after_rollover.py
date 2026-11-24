"""Stage and run one claimed, zero-GRES synchronized measurement of the live fleet."""

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
PARENTS = ('1103606', '1194900')
NAMES = {'collect_training80_busy_pulses.py', 'check_training80_point_clock.py', 'sample_training80_busy.py',
         'training_dispatch_allocation.py', 'ledger.json', 'pci.json'}
HEX = re.compile('[0-9a-f]{64}')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(doc):
    return (json.dumps(doc, sort_keys=True, allow_nan=False) + '\n').encode()


def save(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def read(path, maximum):
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    require(0 < len(raw) <= maximum, 'wrong artifact length: ' + path.name)
    return raw


def query(*argv):
    raw = subprocess.check_output(argv, stdin=subprocess.DEVNULL, timeout=7)
    require(len(raw) < 2**20, 'scheduler query overflow')
    return raw.decode().splitlines()


def fields(line):
    return dict(part.split('=', 1) for part in line.split() if '=' in part)


def snapshot(ledger):
    groups = ledger['groups']
    exact = {}
    no_numbered = {}
    for parent in PARENTS:
        [job] = list(map(fields, query('scontrol', '-o', 'show', 'job', parent)))
        expected = {node for row in groups.values() for node in row['allocations'][parent]}
        actual = query('scontrol', 'show', 'hostnames', job['NodeList'])
        require(job['JobState'] == 'RUNNING' and job['JobId'] == parent and job['NumNodes'] == '40'
                and job['UserId'].endswith(f'({os.getuid()})') and 'gres/dcu=320' in job['AllocTRES'].split(',')
                and len(actual) == len(set(actual)) == 40 and set(actual) == expected,
                'parent allocation or ownership changed')
        live = {}
        used = {}
        for row in map(fields, query('scontrol', '-o', 'show', 'step', parent)):
            step = row.get('StepId', '')
            if not re.fullmatch(parent + r'\.[0-9]+', step) or row.get('State') != 'RUNNING':
                continue
            gres = re.search(r'(?:^|,)gres/dcu=([0-9]+)(?:,|$)', row.get('TRES', ''))
            if not gres or int(gres[1]) == 0:
                continue
            expr = row['NodeList']
            nodes = query('scontrol', 'show', 'hostnames', expr) if re.search(r'[\[,]', expr) else [expr]
            require(len(set(nodes)) == len(nodes) and set(nodes) <= expected,
                    'numbered compute step outside owned nodes')
            cards = int(gres[1])
            require(0 < cards <= 8 * len(nodes), 'impossible compute allocation')
            require(cards == 8 * len(nodes) or len(nodes) == 1, 'cannot assign partial multi-node GPU step per node')
            per_node = cards // len(nodes)
            for node in nodes:
                require(0 <= used.get(node, 0) + per_node <= 8, 'ambiguous overlapping GPU allocation')
                used[node] = used.get(node, 0) + per_node
            require(step not in live, 'duplicated numbered step')
            live[step] = {'nodes': sorted(nodes), 'cards': cards}
        exact[parent] = live
        no_numbered[parent] = sorted(expected - used.keys())
    return {'steps': exact, 'no_numbered': no_numbered,
            'numbered_cards': sum(row['cards'] for p in exact.values() for row in p.values())}


def collect(envelope):
    require(set(envelope) == {'claim', 'publication', 'files'}, 'wrong caller envelope')
    claim_raw = base64.b64decode(envelope['claim'], validate=True)
    claim = json.loads(claim_raw)
    request = claim['request']
    require(re.fullmatch('[0-9a-f]{32}', request) and re.fullmatch('[0-9a-f]{40}', envelope['publication'])
            and claim['points'] == 5 and claim['nodes'] == 80 and claim['GPU_requested'] == 0
            and claim['card_samples'] == 3200 and claim['max_span_ms'] == 250
            and claim['parents'] == list(PARENTS) and claim['replay_allowed'] is False,
            'claim scope differs')
    require(set(envelope['files']) == set(claim['files_SHA256']) == NAMES, 'wrong input file set')
    files = {name: base64.b64decode(value, validate=True) for name, value in envelope['files'].items()}
    require(all(0 < len(raw) < 256 * 1024 and HEX.fullmatch(claim['files_SHA256'][name])
                and sha(raw) == claim['files_SHA256'][name] for name, raw in files.items()), 'frozen input bytes differ')
    stage = ROOT / 'incoming' / ('training80-sync-after-rollover-' + request)
    output = ROOT / 'results' / ('training80-sync-after-rollover-' + request)
    require(not os.path.lexists(stage) and not os.path.lexists(output), 'original measurement namespace exists')
    before = snapshot(json.loads(files['ledger.json']))
    stage.mkdir(mode=0o700)
    for name, raw in files.items():
        save(stage / name, raw)
    intent = {'request': request, 'publication': envelope['publication'], 'claim_SHA256': sha(claim_raw),
              'GPU_numbered_before': before, 'replay_allowed': False}
    save(stage / 'dispatch-intent.json', dump(intent))
    command = [sys.executable, '-E', '-s', '-S', '-B', '-u', str(stage / 'collect_training80_busy_pulses.py'),
               'collect', '--request', request, '--ledger', str(stage / 'ledger.json'),
               '--pci-pins', str(stage / 'pci.json'), '--output-dir', str(output),
               '--points', '5', '--max-span-ms', '250']
    result = {'request': request, 'publication': envelope['publication'], 'GPU_numbered_before': before['numbered_cards'],
              'missing_numbered_before': before['no_numbered'], 'state': 'SAMPLER_UNVERIFIED'}
    try:
        with (stage / 'sampler.stdout').open('xb') as out, (stage / 'sampler.stderr').open('xb') as err:
            done = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=74, check=False)
        result['sampler_rc'] = done.returncode
        raw = read(output / 'outcome.json', 512 * 1024)
        actual = json.loads(raw)
        require(actual['request'] == request and actual['ledger_SHA256'] == claim['files_SHA256']['ledger.json']
                and actual['pci_pins_SHA256'] == claim['files_SHA256']['pci.json'], 'original outcome identity differs')
        result.update(original_SHA256=sha(raw), original_bytes=len(raw),
                      sampler={k: v for k, v in actual.items() if k != 'points'},
                      points=[{k: v for k, v in point.items() if k not in ('worker_reports', 'missing_PCI_by_node')}
                              for point in actual['points']])
        after = snapshot(json.loads(files['ledger.json']))
        result.update(GPU_numbered_after=after['numbered_cards'], missing_numbered_after=after['no_numbered'],
                      same_numbered_GPU_membership=before['steps'] == after['steps'])
        result['state'] = 'COMPLETE_WINDOW' if actual['complete'] and done.returncode == 0 else 'INCOMPLETE_WINDOW'
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        result['error_type'] = type(error).__name__
    finally:
        save(stage / 'dispatch-result.json', dump(result))
    return result


if __name__ == '__main__':
    os.umask(0o077)
    raw = sys.stdin.buffer.read((1 << 20) + 1)
    require(0 < len(raw) <= 1 << 20, 'measurement envelope exceeds bound')
    result = collect(json.loads(raw))
    print(dump(result).decode(), end='', flush=True)
    raise SystemExit(int(result['state'] != 'COMPLETE_WINDOW'))
