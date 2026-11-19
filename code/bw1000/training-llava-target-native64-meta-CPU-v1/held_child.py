"""Terminate/reap only the Popen object held by this new-session supervisor."""
import os
import signal
import subprocess
import time


def cleanup_held(child, anchor, deadline):
    result = dict(PID=child.pid if child is not None else None, termination_needed=False,
                  owned_group_signaled=False, reaped=child is None, returncode=None)
    if child is None:
        return result
    if anchor['PID'] != child.pid or anchor['PGID'] != child.pid or anchor['start_new_session'] is not True:
        raise ValueError('held child anchor differs; never signal another process')
    # Do not poll/reap before this check: with returncode None the held child has
    # not been reaped, so its PID cannot identify a replacement process.
    unreaped = child.returncode is None
    same_owned_group = False
    if unreaped and os.name == 'posix':
        try:
            same_owned_group = os.getpgid(child.pid) == anchor['PGID']
        except ProcessLookupError:
            pass
    def send(kind):
        if same_owned_group:
            try:
                os.killpg(anchor['PGID'], kind)
                result['owned_group_signaled'] = True
            except ProcessLookupError:
                pass
        elif child.returncode is None:
            # The held Popen API checks its own unreaped child. No foreign PID
            # or unverified group is used even if /proc snapshot failed.
            if kind == signal.SIGTERM:
                child.terminate()
            else:
                child.kill()
    if unreaped:
        result['termination_needed'] = True
        send(signal.SIGTERM)
        if same_owned_group:
            # Keep the root unreaped throughout the grace period. Its numeric
            # PID/PGID remains anchored even if it becomes a zombie while a
            # descendant ignores TERM; do not release that identity with poll.
            until = min(deadline, time.time() + 10.0)
            while time.time() < until:
                time.sleep(max(0, min(.05, until - time.time())))
            send(signal.SIGKILL)
    try:
        child.wait(timeout=max(0, min(10.0, deadline - time.time())))
    except subprocess.TimeoutExpired:
        send(signal.SIGKILL)
        try:
            child.wait(timeout=max(0, min(1.0, deadline - time.time())))
        except subprocess.TimeoutExpired:
            pass
    result['returncode'] = child.returncode
    result['reaped'] = child.returncode is not None
    return result
