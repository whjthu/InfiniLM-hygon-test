"""Run exactly one guarded C4 relink inside the claimed zero-GPU Slurm step."""

import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from verify_deep_c4_direct_history_link import OUTPUT, audit, require


PARENT = '1103606'
NODE = 'f13r2n01'
OLD_STEPS = {'156', '487'}


def bind_step(environ, hostname, now):
    step = environ.get('SLURM_STEP_ID', '')
    require(environ.get('SLURM_JOB_ID') == PARENT and re.fullmatch('[1-9][0-9]*', step)
            and step not in OLD_STEPS and hostname == NODE and environ.get('SLURM_CPUS_PER_TASK') == '4',
            'unexpected CPU relink step or placement')
    require(not environ.get('SLURM_STEP_GPUS')
            and all(environ.get(key) == '-1' for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
                                                        'ROCR_VISIBLE_DEVICES')),
            'CPU relink step exposes GPU devices')
    request = environ.get('TUNING_CPU_REQUEST', '')
    publication = environ.get('TUNING_CPU_PUBLICATION', '')
    manifest = environ.get('TUNING_CPU_MANIFEST_SHA', '')
    require(re.fullmatch('[0-9a-f]{32}', request) and re.fullmatch('[0-9a-f]{40}', publication)
            and re.fullmatch('[0-9a-f]{64}', manifest), 'missing published CPU request binding')
    epoch = float(environ.get('TUNING_CPU_EPOCH', 'nan'))
    deadline = float(environ.get('TUNING_CPU_DEADLINE', 'nan'))
    require(all(math.isfinite(x) for x in (epoch, deadline, now)) and deadline == epoch + 300
            and epoch <= now < deadline - 7, 'invalid original CPU epoch or expired deadline')
    return dict(step=f'{PARENT}.{step}', node=NODE, request=request, publication=publication,
                source_manifest_SHA256=manifest, epoch=epoch, deadline=deadline)


def deadline_clock(binding, wall=time.time, monotonic=time.monotonic):
    start = monotonic()
    wall_at_start = wall()
    limit = min(265.0, binding['deadline'] - wall_at_start - 7.0)
    def remaining():
        elapsed = monotonic() - start
        left = min(limit - elapsed, binding['deadline'] - wall() - 7.0)
        if not math.isfinite(left) or elapsed < 0 or left <= 0:
            raise TimeoutError('original CPU relink deadline exhausted')
        return left
    return remaining


def save(path, record):
    with path.with_suffix('.tmp').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    path.with_suffix('.tmp').rename(path)


def run_once(output, binding, preflight, launch_linker, verify, remaining):
    remaining()
    output.mkdir(mode=0o700)
    identity = dict(binding, new_translation_units=0, actual_GPU_calls=0,
                    checkpoint_weight_reads=0, runtime_provider_verified=False,
                    GPU_GO=False, Model_GO=False)
    save(output / 'intent.json', dict(identity, replay_allowed=False, guarded_probe_object_reuse_planned=True))
    receipt = dict(identity, completed=False, passed=False, linker_attempts=0, guarded_probe_object_reused=False)
    try:
        plan = preflight()
        save(output / 'link-plan.json', plan)
        seconds = min(90.0, remaining() - 35.0)
        require(seconds >= 2, 'insufficient original budget for linker and final ELF audit')
        save(output / 'link-attempt.json', dict(request=binding['request'], linker_timeout_seconds=seconds,
                                               original_deadline=binding['deadline'], replay_allowed=False))
        receipt['linker_attempts'] = 1
        rc = launch_linker(plan['linker_argv'], seconds)
        receipt['linker_returncode'] = rc
        require(rc == 0, 'guarded probe linker failed')
        receipt['guarded_probe_object_reused'] = True
        remaining()
        audited = verify()
        require(audited['relinked_probe_passive_verified'] is True
                and all(audited[key] == plan[key] for key in ('input_SHA256', 'expected_DT_NEEDED',
                                                            'linker_argv', 'candidate_receipt_SHA256')),
                'original input or linkage pins drifted during linking')
        receipt.update(passed=True, relinked_probe_SHA256=audited['relinked_probe_SHA256'],
                       expected_DT_NEEDED=audited['expected_DT_NEEDED'])
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        receipt.update(error_kind=type(error).__name__, failed_check=str(error)[:240])
    finally:
        receipt['completed'] = True
        save(output / 'CPU-result.json', receipt)
    return receipt


def main():
    binding = bind_step(os.environ, socket.gethostname(), time.time())
    remaining = deadline_clock(binding)
    output = Path(OUTPUT)

    def launch(argv, seconds):
        command = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=5s', f'{seconds:.3f}s', *argv]
        with (output / 'link.stdout').open('xb') as stdout, (output / 'link.stderr').open('xb') as stderr:
            return subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                  timeout=seconds + 8, check=False).returncode

    def inspect(verify=False):
        return audit(Path('/refactor'), verify, remaining=remaining, output_reserved=True)

    result = run_once(output, binding, inspect, launch, lambda: inspect(True), remaining)
    print(json.dumps(result, allow_nan=False), flush=True)
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
