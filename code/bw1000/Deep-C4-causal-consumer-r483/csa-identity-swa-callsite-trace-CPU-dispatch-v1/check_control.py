"""Exercise the co-resident Slurm gate without host mutations or job submission."""

import argparse
import base64
import json
from pathlib import Path
from unittest.mock import patch

import control


def expect_rejection(call, reason):
    try:
        call()
    except ValueError as error:
        assert reason in str(error), str(error)
    else:
        raise AssertionError(reason)


def check(stage):
    manifest = (stage / 'STAGE_FILES.json').read_bytes()
    pins = json.loads(manifest)['files_SHA256']
    b64 = lambda value: base64.b64encode(value).decode()
    envelope = {'manifest': b64(manifest), 'files': {name: b64((stage / name).read_bytes()) for name in pins}}
    assert len(control.stage_files(envelope)) == 11
    damaged = dict(envelope, files={**envelope['files'], 'inside.sh': b64(b'other')})
    expect_rejection(lambda: control.stage_files(damaged), 'stage member changed')

    now = control.STANDBY_START + 1800
    cards = [f'0000:{index:02x}:00.0' for index in range(8)]
    worker = {'pci': cards, 'epoch': control.STANDBY_START, 'worker_pid': 1710860, 'synthetic': True, 'step': control.STANDBY_STEP}
    records = {'worker.json': worker, 'expected.json': {'seconds': control.STANDBY_SECONDS, 'request': control.STANDBY_REQUEST}}
    records.update({f'card-{index}.json': {'pci': pci, 'epoch': now - 4, 'calls': 128, 'active': True} for index, pci in enumerate(cards)})
    standby = (f'StepId={control.STANDBY_STEP} Name={control.STANDBY_LABEL} State=RUNNING NodeList={control.NODE} '
               f'TRES={control.STANDBY_TRES}')
    noise = f'StepId={control.PARENT}.188 Name=other State=TIMEOUT NodeList={control.NODE} TRES={control.STANDBY_TRES}'
    rows = [noise, standby]

    def fixtures(*argv):
        if argv[2:4] == ('show', 'job'):
            return [f'JobId={control.PARENT} JobState=RUNNING NumNodes=40 UserId=test(42) AllocTRES=cpu=5120,gres/dcu=320 NodeList=f[0-39]']
        if argv[2:4] == ('show', 'step'):
            return rows
        assert argv[:3] == ('scontrol', 'show', 'hostnames')
        return [control.NODE] + [f'fixture-{i}' for i in range(39)]

    with patch.object(control.os, 'getuid', return_value=42, create=True), patch.object(control.os.path, 'lexists', return_value=False), \
         patch.object(control, 'read', side_effect=lambda path, *_: control.encode(records[path.name])):
        assert control.admitted_standby(fixtures, wall=lambda: now)['StepId'] == control.STANDBY_STEP
        rows.append(f'StepId={control.PARENT}.444 Name=collision State=RUNNING NodeList={control.NODE} TRES=cpu=2,mem=1G,node=1')
        expect_rejection(lambda: control.admitted_standby(fixtures, wall=lambda: now), 'known standby')
        rows.pop()
        records['card-3.json']['epoch'] -= 50
        expect_rejection(lambda: control.admitted_standby(fixtures, wall=lambda: now), 'fresh eight-card')
        for key in records.keys() - {'worker.json', 'expected.json'}:
            records[key]['epoch'] = control.STANDBY_START + control.STANDBY_SECONDS - 180
        expect_rejection(lambda: control.admitted_standby(fixtures, wall=lambda: records['card-3.json']['epoch']), 'not enough time')
    args = control.command(1795500600.0, '0' * 64, Path('/original/outer.sh'))
    assert '--gres=none' in args and '--cpus-per-task=4' in args and '--immediate=5' in args and '--overlap' not in args
    print(json.dumps({'stage_members': len(pins), 'local_guard_cases_passed': 5, 'target_calls': 0}, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', type=Path)
    check(parser.parse_args().stage)
