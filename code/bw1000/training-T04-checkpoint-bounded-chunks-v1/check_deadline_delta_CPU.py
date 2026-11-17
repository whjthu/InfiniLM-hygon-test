"""Only newly changed alarm/held-child/admission boundaries; no remote or payload."""
import ast
import json
from pathlib import Path
import subprocess
import sys

D = Path(__file__).resolve().parent


def extract(name, names, namespace=None):
    tree = ast.parse((D/name).read_bytes())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    result = {} if namespace is None else namespace
    exec(compile(ast.Module(body=nodes, type_ignores=[]), name, 'exec'), result)
    return result


def main():
    valid, rejected = [], []
    source = extract('relay_once.py', {'remaining_seconds', 'kill_children'})
    left, kill = source['remaining_seconds'], source['kill_children']
    assert left(100, 10, 90, 113, 22) == 62
    valid.append('Git-admission-spent13s-not-reset')
    for label, wall, mono in [('Popen-registration-after-timer', 180, 90),
                              ('wall-clock-jump-forward', 195, 11),
                              ('wall-clock-backwards-monotonic-exhausted', 0, 101)]:
        try:
            left(100, 10, 90, wall, mono)
        except TimeoutError:
            rejected.append(label)
        else:
            raise AssertionError(label)

    class Child:
        def __init__(self, pid, race=False):
            self.pid, self.race, self.killed = pid, race, False
        def poll(self):
            return None
        def kill(self):
            if self.race:
                raise ProcessLookupError('new exit race')
            self.killed = True
    first, second = Child(1, True), Child(2)
    forced, errors = set(), []
    kill({'first': first, 'second': second}, forced, errors)
    assert second.killed and forced == {1, 2} and errors == ['owned-child-kill:ProcessLookupError']
    valid.append('first-child-kill-race-still-cleans-second')

    held = subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c', 'import time; time.sleep(30)'],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    owned = {'new-held-child': held}
    forced, errors = set(), []
    try:
        left(100, 10, 90, 180, 90)
    except TimeoutError:
        kill(owned, forced, errors)
    finally:
        if held.poll() is None:
            held.kill()
        held.wait(timeout=5)
    assert held.pid in forced and held.returncode is not None and not errors
    valid.append('actual-new-local-held-Popen-reaped-after-expired-registration')

    class Clock:
        wall, mono = 185, 86
        def time(self):
            return self.wall
        def monotonic(self):
            return self.mono
    class Signals:
        SIGALRM, ITIMER_REAL = 14, 0
        def __init__(self):
            self.handlers, self.timers = [], []
        def signal(self, number, handler):
            self.handlers.append((number, handler))
        def setitimer(self, number, seconds):
            self.timers.append((number, seconds))
    class Exit:
        def _exit(self, code):
            raise SystemExit(code)
    clock, signals = Clock(), Signals()
    endpoint = extract('endpoint.py', {'hard_exit', 'work_alarm'},
        dict(CONTEXT={'deadline': 190}, HARD_MONO_END=90, time=clock, signal=signals, os=Exit()))
    try:
        endpoint['work_alarm']()
    except TimeoutError:
        pass
    else:
        raise AssertionError('work alarm must stop work')
    assert signals.handlers == [(14, endpoint['hard_exit'])] and signals.timers == [(0, 4)]
    valid.append('work-alarm-rearms-only-original-4s-tail-before-raising')
    clock.wall = 191
    try:
        endpoint['work_alarm']()
    except SystemExit as error:
        assert error.code == 124
        rejected.append('past-original-deadline-hard-exit124')
    else:
        raise AssertionError('expired tail extended')
    print(json.dumps(dict(passed=True, valid=valid, rejected=rejected, valid_count=len(valid),
        rejected_count=len(rejected), actual_local_child_reaped=True, remote_calls=0,
        real_POSIX_signal_test=False, handler_branch_with_fake_signal_and_clock=True)))


if __name__ == '__main__':
    main()
