"""Identity-check the held evaluator child, then exec unchanged inference worker."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

D = Path(__file__).resolve().parent


def need(ok, code):
    if not ok:
        raise ValueError(code)


def option(name):
    need(sys.argv.count(name) == 1, 'ONE_ARGUMENT')
    return sys.argv[sys.argv.index(name) + 1]


def main():
    path = Path(option('--admission')); raw = path.read_bytes()
    need(len(raw) == int(option('--admission-bytes')) and hashlib.sha256(raw).hexdigest() == option('--admission-sha'), 'ADMISSION_PIN')
    a = json.loads(raw); owner = a['owned_child']; parent = a['supervisor']
    row = Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()
    need(owner['PID'] == os.getpid() == owner['PGID'] == os.getpgrp()
         and owner['startticks'] == int(row[19]) and int(row[1]) == parent['PID']
         and owner['UID'] == os.getuid() and owner['host'] == socket.gethostname()
         and owner['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'EXACT_HELD_OWNER')
    parent_row = (Path('/proc') / str(parent['PID']) / 'stat').read_text().rsplit(') ', 1)[1].split()
    need(parent['startticks'] == int(parent_row[19]), 'SAME_PARENT_STARTTICKS')
    source = D / 'bounded_eval.py'
    need(hashlib.sha256(source.read_bytes()).hexdigest() == a['parent_source_SHA256']
         and os.fsencode(source) in (Path('/proc') / str(parent['PID']) / 'cmdline').read_bytes().split(b'\0'), 'ACTUAL_OWNED_PARENT_SOURCE')
    need(os.environ['CUDA_VISIBLE_DEVICES'] == a['selected_GPU_UUID']
         and sorted(os.sched_getaffinity(0)) == a['CPU_affinity']
         and len(a['CPU_affinity']) == 4 and min(a['CPU_affinity']) >= 2,
         'ACTUAL_ONE_GPU_FOUR_CPU_AFFINITY')
    need(a['epoch'] <= time.time() < a['deadline'] == a['epoch'] + 820, 'SAME_NATIVE_EPOCH')
    need(not Path(option('--output')).exists(), 'NATIVE_OUTPUT_MUST_INITIALly_NOT_EXIST')
    os.execv(sys.executable, [sys.executable, '-B', str(D / 'core/infer_confirmation25.py'), *sys.argv[1:]])


if __name__ == '__main__':
    main()
