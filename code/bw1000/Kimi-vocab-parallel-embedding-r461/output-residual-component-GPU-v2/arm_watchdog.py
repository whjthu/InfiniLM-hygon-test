"""Bound the startup and native phases of exactly one owned arm process."""
import json
import subprocess
import time


def wait(proc, startup_receipt, started, whole_deadline, startup_seconds, native_seconds):
    limit = min(started + startup_seconds, whole_deadline)
    startup = None
    while startup is None:
        try:
            startup = json.loads(startup_receipt.read_bytes())
        except FileNotFoundError:
            rc = proc.poll()
            if rc is not None:
                return rc, None
            if time.monotonic() >= limit:
                raise TimeoutError("worker startup exceeded its bound")
            time.sleep(.1)
    end = startup.get("ready_monotonic_ns")
    if (startup.get("PID") != proc.pid or type(end) is not int
        or not started <= end * 1e-9 <= min(time.monotonic(), limit)):
        raise ValueError("unexpected startup receipt identity or deadline")
    native_bound = min(end * 1e-9 + native_seconds, whole_deadline)
    rc = proc.poll()
    if rc is None:
        if time.monotonic() >= native_bound:
            raise TimeoutError("worker native phase exceeded its bound")
        try:
            rc = proc.wait(timeout=native_bound - time.monotonic())
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("worker native phase exceeded its bound") from exc
    return rc, startup
