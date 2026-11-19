"""Durable process boundaries for the original ctypes caller."""
import os
import stat
import sys
import time


def mark(name):
    print(f"kimi-stage python {mark.sequence} {time.monotonic_ns()} {name}", file=sys.stderr, flush=True)
    mark.sequence += 1
    fd = sys.stderr.fileno()
    if stat.S_ISREG(os.fstat(fd).st_mode):
        os.fsync(fd)


mark.sequence = 0
