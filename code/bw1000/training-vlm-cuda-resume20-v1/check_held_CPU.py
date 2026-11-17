"""Only new held-child cleanup boundaries; no old checks/provider imports."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

D = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location('actual_held', D / 'held_child.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    creation = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {'start_new_session': True}
    child = subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c', 'import time; time.sleep(30)'], **creation)
    try:
        result = module.cleanup_held(child, dict(PID=child.pid, PGID=child.pid, start_new_session=True), time.time() + 2)
        assert result['termination_needed'] and result['reaped'] and child.returncode is not None
    finally:
        if child.poll() is None:
            child.kill(); child.wait(timeout=2)
    events = []
    class FakeChild:
        pid = 12345; returncode = None
        def terminate(self):
            events.append('held.terminate')
        def kill(self):
            events.append('held.kill')
        def wait(self, timeout):
            events.append('held.reap'); self.returncode = -15; return self.returncode
    anchor = dict(PID=12345, PGID=12345, start_new_session=True)
    module.signal = SimpleNamespace(SIGTERM=15, SIGKILL=9)
    fake_os = SimpleNamespace(name='posix', getpgid=lambda pid: 999,
                              killpg=lambda *args: events.append('FOREIGN_GROUP_SIGNAL'))
    module.os = fake_os
    foreign = module.cleanup_held(FakeChild(), anchor, time.time() + 1)
    assert foreign['reaped'] and events == ['held.terminate', 'held.reap']
    events.clear()
    fake_os.getpgid = lambda pid: pid
    fake_os.killpg = lambda pid, kind: events.append(('OWNED_GROUP', pid, kind))
    # Expired cleanup reserve forces immediate TERM+KILL, still under the same
    # held/unreaped root anchor; no fake clock or budget reset.
    group = module.cleanup_held(FakeChild(), anchor, time.time() - 1)
    assert group['reaped'] and events == [('OWNED_GROUP', 12345, 15), ('OWNED_GROUP', 12345, 9), 'held.reap']
    events.clear()
    try:
        module.cleanup_held(FakeChild(), dict(anchor, PGID=999), time.time() + 1)
    except ValueError:
        pass
    else:
        raise AssertionError('foreign anchor accepted')
    assert not events
    return dict(passed=True, actual_owned_child_terminated_and_reaped=True,
                snapshot_registry_was_empty=True, local_platform=os.name,
                POSIX_group_branch_fixture_only=True, foreign_group_not_signaled=True,
                expired_reserve_no_extension=True, forged_anchor_rejected=True,
                actual_GPU=False, actual_provider_import=False)


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
