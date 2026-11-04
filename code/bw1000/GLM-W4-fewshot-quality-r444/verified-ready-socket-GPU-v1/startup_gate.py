"""External keeper startup bound, using real Run records, never fake READY.

No thread, timer, model import, Native call or Slurm submission is created here.
The existing keeper loop calls observe before it can start the client.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time


class StartupTimeout(TimeoutError):
    pass


class StartupGate:
    def __init__(self, run):
        self.run = run
        self.first_all16_ready_unix = None
        self.accepted = False
        self.blocked = False

    def observe(self, now):
        if self.blocked:
            raise StartupTimeout('Startup failure latched: client forbidden')
        self.run.raise_if_failed()
        if (self.run.path / 'STOP.json').exists():
            self.run.read('STOP.json')
            self.blocked = True
            raise StartupTimeout('STOP before public READY: client forbidden')
        if self.accepted:
            return True
        stacks = list(self.run.path.glob('startup-stack-*.txt'))
        assert len(stacks) <= 1, 'Only the single emitter process owns this watchdog'
        if any(path.stat().st_size > 8 * 1024 ** 2 for path in stacks):
            self.blocked = True
            raise StartupTimeout('Startup diagnostic trace exceeded8MiB; preserve original and drain')
        names = ['ready-%d-%d.json' % (s, r) for s in range(2) for r in range(8)]
        if all((self.run.path / name).is_file() for name in names):
            # Files are atomically published. This invokes exactly the original
            # identity, load, config and 32-disjoint-mask checks, not mere exists.
            records = self.run.all_ready()
            assert len(records) == 16
            if self.first_all16_ready_unix is None:
                self.first_all16_ready_unix = now
        epoch = self.run.config['gpu_start_epoch']
        expired_epoch = now >= epoch + 2700
        expired_ready = (self.first_all16_ready_unix is not None and
                         now >= self.first_all16_ready_unix + 120)
        # Evaluate before accepting a late MODEL_READY: STOP cannot unlock an
        # already failed startup and cause the keeper to launch a client.
        if expired_epoch or expired_ready:
            self.blocked = True
            raise StartupTimeout('Startup bound exceeded: epoch2700=%s all16ready120=%s' %
                                 (expired_epoch, expired_ready))
        ready = self.run.path / 'MODEL_READY.json'
        if ready.is_file():
            value = self.run.read(ready.name)
            assert self.first_all16_ready_unix is not None
            assert value['Model_READY'] and value['actual_owner_count'] == 16
            self.accepted = True
        return self.accepted

    def state(self):
        return dict(first_all16_ready_unix=self.first_all16_ready_unix,
                    startup_accepted=self.accepted, client_launch_blocked=self.blocked,
                    startup_epoch_deadline_unix=self.run.config['gpu_start_epoch'] + 2700,
                    startup_ready_deadline_unix=(None if self.first_all16_ready_unix is None
                                                else self.first_all16_ready_unix + 120))


def seal_startup_records(folder, output):
    """Original small diagnostics only; raw large/active model reports untouched.

    Original files stay in the run directory. Capture EOF snapshots with bytes
    and SHA; an in-progress repeating stack is explicitly a partial snapshot.
    """
    folder, output = Path(folder), Path(output)
    names = sorted({*folder.glob('startup-*.jsonl'), *folder.glob('startup-stack-*.txt')})
    assert len(names) <= 128, 'Unexpected diagnostic file count'
    rows = []
    total_read = 0
    for path in names:
        size = path.stat().st_size
        if size > 16 * 1024 ** 2 or total_read + size > 32 * 1024 ** 2:
            rows.append(dict(path=str(path),bytes_at_stat=size,not_read='exceeds16MiB',
                             original_retained=True))
            continue
        raw = path.read_bytes()
        total_read += len(raw)
        rows.append(dict(path=str(path),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),
                         rawtext=raw.decode('utf-8', errors='replace'),
                         snapshot_may_end_in_partial_line=True,original_retained=True))
    result = dict(observed_unix=time.time(),files=rows,whole_model_reports_read=False)
    with output.open('x') as out:
        json.dump(result,out,indent=2);out.write('\n')
    return dict(path=str(output),bytes=output.stat().st_size,
                SHA256=hashlib.sha256(output.read_bytes()).hexdigest(),files=len(rows))


def drain_after_stop(proc, exact, label, recorded_step, *, clock=time.monotonic,
                     sleep=time.sleep):
    """Give the original STOP a real60s idle drain before any TERM/KILL.

    Unknown accounting never means normal exit; transient observations cannot
    restart the timer. This helper sends no signal and does not create a client.
    """
    end = clock() + 60
    result = dict(normal_GPU_STOP_drain=False,observations=[],signals=[])
    while True:
        try:
            row = exact(label, timeout=max(1,min(5,end-clock())))
            if row:
                assert row[0] == recorded_step, 'Owned drain step mismatch'
            result['observations'].append(dict(row=row,srun_rc=proc.poll()))
            if row and row[2:4] == ['COMPLETED','0:0'] and proc.poll() == 0:
                result['normal_GPU_STOP_drain'] = True
                return result
        except (OSError, TimeoutError, subprocess.TimeoutExpired,
                subprocess.CalledProcessError) as error:
            result['observations'].append(dict(error=repr(error),srun_rc=proc.poll()))
        if clock() >= end:
            return result
        sleep(min(1,max(0,end-clock())))
