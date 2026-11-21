"""Exact replacement functions for the frozen controller namespace; never launch alone."""


def step_command(phase, command, *, seconds=None):
    devices, cpus, memory, nominal = PHASES[phase]
    seconds = nominal if seconds is None else seconds
    need(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= nominal,
         'invalid original phase step budget')
    # Slurm rounds requested seconds upward to whole minutes. Request whole
    # minutes rounded DOWN, and refuse a zero-minute/unlimited request.
    minutes = math.floor(seconds / 60)
    need(minutes >= 1, 'insufficient remaining budget for one bounded Slurm minute')
    return ['srun', '--jobid=' + PARENT, '--exact', '--immediate=5', '--cpu-bind=none', '-N1', '-n1',
            '-c' + str(cpus), f'--mem={memory}M', '--gres=' + (f'dcu:{devices}' if devices else 'none'),
            '-w', NODE, '-t', str(minutes), '--job-name=' + LABEL + '-' + phase, *command]


def execute_step(phase, command, elapsed, epoch):
    sys.path.insert(0, str(ACCOUNTING_HELPER))
    from slurm_accounting import wait_terminal
    label = LABEL + '-' + phase
    need(accounting(label) is None, 'original Slurm label is already present')
    used = elapsed()
    need(type(used) in (int, float) and math.isfinite(used) and used >= 0
         and type(epoch) in (int, float) and math.isfinite(epoch) and epoch > 0,
         'nonfinite or invalid original phase clock')
    cap = PHASES[phase][3] + 5
    if phase == 'gpu':
        need(used < NATIVE - 60, 'insufficient original native budget')
        cap = min(cap, PHASES[phase][3] - used)
    else:
        need(used + cap + 26 < WHOLE, 'insufficient CPU step and accounting budget')
    # This is the ACTUAL argv passed to subprocess, not a separate advisory plan.
    # Reserve the existing --immediate=5 interval within the client cap.
    seconds = min(PHASES[phase][3], cap - 5)
    argv = step_command(phase, command, seconds=seconds)
    result = dict(phase=phase, rc=None, terminal_visible=False, row=None, passed=False, observations=[],
                  budget=dict(observed_elapsed_seconds=used, client_timeout_seconds=cap,
                              actual_Slurm_time_minutes=int(argv[argv.index('-t')+1]),
                              requested_step_seconds=60*int(argv[argv.index('-t')+1]),
                              launch_interval_reserve_seconds=5, original_epoch=epoch,
                              original_whole_deadline=epoch+WHOLE))
    try:
        with (RUN / (phase + '.log')).open('xb') as log:
            try:
                result['rc'] = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                              timeout=cap).returncode
            except subprocess.TimeoutExpired:
                result['client_timeout'] = True
        deadline = time.monotonic() + min(24, max(0, WHOLE - elapsed() - 9))
        result.update(wait_terminal(lambda: accounting(label), deadline, max_observations=8))
        row = result['row']
        result['passed'] = bool(result['rc'] == 0 and row and REQUEST_PATTERN.fullmatch(row[0])
             and row[2:4] == ['COMPLETED', '0:0'] and row[5] == NODE and allocated_dcu(row[6]) == PHASES[phase][0])
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result['observation_error'] = type(error).__name__
        raise
    finally:
        write(RUN / (phase + '-accounting.json'), dict(epoch=epoch, **result))
    return result
