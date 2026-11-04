"""One shared run namespace; scalar/JSON/base64 only, never NativeRequest IPC."""
import json
import os
from pathlib import Path
import re
import time
from .cache_policy_r452 import global_metadata,for_config


def publish_once(path, value):
    path = Path(path)
    raw = json.dumps(value, indent=2, allow_nan=False) + '\n'
    # Unique writes are completed to a private file, then published without
    # overwrite. Consumers never parse an intentional partial final file.
    temp = path.with_name(path.name + '.writing-' + str(os.getpid()))
    with temp.open('x') as out:
        out.write(raw);out.flush();os.fsync(out.fileno())
    try: os.link(temp, path)
    finally: temp.unlink()


class Run:
    def __init__(self, config):
        self.config = dict(config)
        assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', config['run'])
        base = Path(config['runtime_directory']).resolve()
        self.path = base / config['run']
        assert self.path.resolve().is_relative_to(base)
        self.path.mkdir(parents=True, exist_ok=True)

    def identity(self):
        c = self.config
        return dict({k:c[k] for k in ['run', 'actual_step', 'source_manifest_SHA256','provider_config_SHA256',
                                'gpu_start_epoch', 'Native_seconds']},cache_policy=global_metadata(c))

    def check(self, row):
        assert row['identity'] == self.identity(), 'stale/foreign run record'
        return row

    def write(self, name, data):
        publish_once(self.path / name, dict(identity=self.identity(), **data))

    def read(self, name):
        return self.check(json.loads((self.path / name).read_bytes()))

    def remaining(self):
        c = self.config
        return float(c['gpu_start_epoch']) + int(c['Native_seconds']) - time.time()

    def deadline_check(self, where, minimum=0):
        self.raise_if_failed()
        service = self.config['gpu_start_epoch'] + self.config['service_seconds'] - time.time()
        if min(self.remaining(), service) <= minimum:
            raise TimeoutError('Original Native deadline: ' + where)

    def raise_if_failed(self):
        if (self.path / 'FAILED.json').exists():
            raise RuntimeError('Peer/owner failed: ' + str(self.read('FAILED.json')))

    def fail(self, error, *, stage, rank=None):
        value = dict(error=repr(error), stage=stage, rank=rank, PID=os.getpid())
        try: self.write('FAILED.json', value)
        except FileExistsError: pass

    def wait(self, name, minimum=120):
        while not (self.path / name).exists():
            if minimum == 0:
                self.raise_if_failed()
                if self.remaining() <= 0: raise TimeoutError('Native cleanup visibility deadline: ' + name)
            else:
                self.deadline_check('wait ' + name, minimum)
            time.sleep(.1)
        return self.read(name)

    def stop_requested(self):
        self.raise_if_failed()
        if (self.path / 'STOP.json').exists():
            self.read('STOP.json')
            return True
        return False

    def stop(self):
        try: self.write('STOP.json', dict(normal_stop=True))
        except FileExistsError: self.read('STOP.json')

    def all_ready(self):
        records = [self.wait('ready-%d-%d.json' % (s, r)) for s in range(2) for r in range(8)]
        return self.validate_ready_records(records)

    def validate_ready_records(self, records):
        if type(records) is not list or len(records) != 16:
            raise ValueError('Exactly16 actual ready records required')
        for row in records:
            self.check(row)  # original wait/read identity gate, also for socket records
        for stage in range(2):
            masks = []
            for rank in range(8):
                row = records[8 * stage + rank]
                assert (row['stage'], row['rank']) == (stage, rank)
                assert row['Native_SHA256'] == self.config['Native_SHA256']
                assert row['cache_policy']==for_config(self.config,stage).metadata()
                assert row['config_SHA256'] == self.config['config_SHA256']
                assert row['loaded'] and row['postload_4GiB'] and row['service_agreement']
                actual = row['affinity']['actual_owner_report']['owners']
                mine = [q for q in actual if q['device'] == rank]
                assert len(mine) == 1 and mine[0]['changes'] == 1
                mask = [int(v) for v in mine[0]['verified_mask'].split(',')]
                assert len(mask) == 4
                masks.extend(mask)
            assert len(masks) == len(set(masks)) == 32, 'cross-process owner masks overlap'
        return records
