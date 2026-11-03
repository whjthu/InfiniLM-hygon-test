"""Independent existing owner cleanup actions; preserve the primary error."""


def close_owner(*, ctx, lib, check, sampler, control, events, report, save, primary):
    errors = []
    destroyed = False
    if sampler is not None:
        try:
            sampler.close()
            assert not sampler.thread.is_alive(), 'Owned memory sampler did not stop'
        except BaseException as error: errors.append(('sampler', error))
    if ctx is not None:
        try:
            check(lib.modern_kimi_engine_destroy_r2475(ctx))
            destroyed = True
            report['all8_owned_workers_shutdown_completed'] = True
        except BaseException as error: errors.append(('Native-destroy', error))
    if destroyed and events is not None:
        try: events.after_normal_destroy()
        except BaseException as error: errors.append(('events-after-destroy', error))
    normal = bool(primary is None and report.get('protocol_before_destroy_passed') and destroyed and not errors)
    if normal:
        try: control.collect('after_destroy', {})
        except BaseException as error: errors.append(('four-leader-destroy-ack', error))
    try: control.close()
    except BaseException as error: errors.append(('control-close', error))
    report['cleanup_errors'] = [dict(phase=phase, error=repr(error)) for phase, error in errors]
    report['completed'] = bool(normal and not errors)
    report['phase'] = 'completed' if report['completed'] else 'failed'
    try: save()
    except BaseException as error: errors.append(('final-report', error))
    if primary is not None:
        if hasattr(primary, 'add_note'):
            for phase, error in errors: primary.add_note(phase + ': ' + repr(error))
        return  # Original except re-raises its primary after this finally.
    if errors: raise RuntimeError('Actual Kimi owner cleanup failed: ' + repr(errors)) from errors[0][1]
